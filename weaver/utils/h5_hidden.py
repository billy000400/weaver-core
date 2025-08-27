# h5_writer.py
import h5py
import numpy as np
import torch


class H5AppendWriter:
    def __init__(self, path, compression="lzf", dtype=np.float32):
        self.f = h5py.File(path, "w")
        self.ds = {}            # name -> h5py.Dataset
        self.count = 0
        self.compression = compression
        self.dtype = dtype
        self.f.attrs["layout"] = "(B, T, D)"
    
    def close(self):
        self.f.flush(); self.f.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


    def _mk(self, name, T, D):
        if name in self.ds: return
        maxshape = (None, T, D)
        samples_per_chunk = max(1, 4096 // max(1, T * D))
        self.ds[name] = self.f.create_dataset(
            name, shape=(0, T, D), maxshape=maxshape,
            chunks=(samples_per_chunk, T, D),
            compression=self.compression, dtype=self.dtype
        )

    def _mk_masks(self, T):
        if "masks" in self.f: return
        spc = max(1, 4096 // max(1, T))
        self.f.create_dataset(
            "masks", shape=(0, T), maxshape=(None, T),
            chunks=(spc, T), compression=self.compression, dtype=np.uint8
        )

    @staticmethod
    def _to_btd(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3: raise ValueError(f"need 3D, got {tuple(x.shape)}")
        # accept (B,T,D) or (T,B,D)
        if x.shape[0] < x.shape[1]:  # likely (T,B,D)
            x = x.permute(1,0,2)
        return x.contiguous().to("cpu")

    # inside H5AppendWriter

    def append(self, layer_tensors, mask=None, layer_names=None, pad_value=0.0):
        """
        Accepts hidden states as:
        - dict[str, Tensor]                (names from keys; layers can have different D)
        - list/tuple[Tensor]               (names auto or from layer_names)
        - Tensor (L,B,T,D)                 (split along L)
        - Tensor (B,T,D)                   (single layer)
        Handles inconsistent (T,D) across calls by resizing datasets and padding previous writes.
        """
        import torch
        import numpy as np

        # ---- Normalize to {name -> tensor} ----
        name_to_tensor = {}
        if isinstance(layer_tensors, dict):
            for k, v in layer_tensors.items():
                name_to_tensor[str(k)] = v
        elif isinstance(layer_tensors, (list, tuple)):
            names = layer_names or [f"hidden_{i+1}" for i in range(len(layer_tensors))]
            if len(names) != len(layer_tensors):
                raise ValueError("layer_names length must match number of tensors.")
            for n, t in zip(names, layer_tensors):
                name_to_tensor[str(n)] = t
        elif isinstance(layer_tensors, torch.Tensor):
            if layer_tensors.ndim == 4:  # (L,B,T,D)
                L = layer_tensors.shape[0]
                names = layer_names or [f"hidden_{i+1}" for i in range(L)]
                if layer_names and len(layer_names) != L:
                    raise ValueError("layer_names length must match L in (L,B,T,D).")
                for i, t in enumerate(layer_tensors.unbind(0)):
                    name_to_tensor[str(names[i])] = t
            elif layer_tensors.ndim == 3:  # (B,T,D) single layer
                n = (layer_names[0] if layer_names else "hidden_1")
                name_to_tensor[str(n)] = layer_tensors
            else:
                raise ValueError(f"Unsupported tensor rank: {layer_tensors.ndim}")
        else:
            raise TypeError("layer_tensors must be dict, sequence, or Tensor.")

        if len(name_to_tensor) == 0:
            return

        # ---- Convert each to (B,T,D) on CPU; collect per-layer shapes ----
        # inside H5AppendWriter.append, replace to_btd_cpu() with:
        def to_btd_cpu(x: torch.Tensor, batch_hint=None) -> torch.Tensor:
            """
            Accept:
            - (B,T,D)  -> return (B,T,D)
            - (T,B,D)  -> permute to (B,T,D)
            - (B,D)    -> promote to (B,1,D)
            - (D,)     -> promote to (1,1,D)
            - (T,D)    -> ambiguous; if batch_hint is provided and T == batch_hint,
                            treat as (B,D)->(B,1,D); else treat as (T,D) with B=1 -> (1,T,D)
            """
            if x.ndim == 3:
                # (B,T,D) or (T,B,D)
                return (x.permute(1,0,2) if x.shape[0] < x.shape[1] else x).contiguous().to("cpu")

            if x.ndim == 2:
                # Could be (B,D) or (T,D). Prefer (B,D); use batch_hint if present.
                B_or_T, D = x.shape
                if batch_hint is not None and B_or_T == batch_hint:
                    # interpret as (B,D)
                    return x.unsqueeze(1).contiguous().to("cpu")          # (B,1,D)
                # Heuristic: if batch dim already established elsewhere, match it
                return x.unsqueeze(1).contiguous().to("cpu")              # default assume (B,D)->(B,1,D)

            if x.ndim == 1:
                # (D,) -> (1,1,D)
                return x.unsqueeze(0).unsqueeze(0).contiguous().to("cpu")

            raise ValueError(f"Each layer must be rank 1/2/3; got {tuple(x.shape)}")


        # ... after you build tensors_btd dict, modify the loop that normalizes:
        tensors_btd = {}
        B_ref = None
        for n, t in name_to_tensor.items():
            x = to_btd_cpu(t, batch_hint=B_ref)
            if B_ref is None:
                B_ref = x.shape[0]
            elif x.shape[0] != B_ref:
                # if we promoted a (T,D) with B=1 incorrectly, try flipping using hint:
                if t.ndim == 2 and t.shape[0] == B_ref:  # treat as (B,D)
                    x = t.unsqueeze(1).contiguous().to("cpu")  # (B,1,D)
                else:
                    raise ValueError(f"Batch size mismatch across layers: {x.shape[0]} vs {B_ref} for layer '{n}'")
            tensors_btd[n] = x
        B = B_ref

        # ---- Ensure /masks can grow (T varies batch-to-batch) ----
        # Determine mask shape if provided
        m_np = None
        T_mask = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                m_np = mask.to("cpu").to(torch.uint8).numpy()
            elif isinstance(mask, np.ndarray):
                m_np = mask.astype(np.uint8, copy=False)
            else:
                raise TypeError("mask must be Tensor or numpy array.")
            if m_np.ndim != 2 or m_np.shape[0] != B:
                raise ValueError(f"mask must be (B,T); got {m_np.shape}")
            T_mask = m_np.shape[1]
            self._mk_masks(T_mask)  # create if missing, with at least T_mask
            # If later batches have larger T, we’ll expand below.

        # ---- For each layer, create or expand dataset as needed, then write (with padding) ----
        for n, x in tensors_btd.items():
            Bx, Tx, Dx = x.shape
            self._ensure_ds_shape(n, Tx, Dx)  # creates if missing; expands T/D if needed

            ds = self.ds[n]  # after _ensure_ds_shape, ds has shape (N_so_far, T_ds, D_ds)
            _, T_ds, D_ds = ds.shape

            # Pad along T and/or D if current ds is larger
            if Tx < T_ds or Dx < D_ds:
                pad_T = T_ds - Tx
                pad_D = D_ds - Dx
                # pad to the right in both axes
                x_np = np.pad(
                    x.numpy(),
                    ((0, 0), (0, max(0, pad_T)), (0, max(0, pad_D))),
                    mode="constant",
                    constant_values=pad_value,
                )
                # enforce exact shape in case of mismatch
                if x_np.shape != (Bx, T_ds, D_ds):
                    x_np = x_np.reshape(Bx, T_ds, D_ds)
            else:
                x_np = x.numpy()

            # Append rows
            ds.resize(self.count + B, axis=0)
            ds[self.count:self.count + B, :, :] = x_np

        # ---- Write /masks with dynamic T as well (expand columns if needed) ----
        if m_np is not None:
            dm = self.f["masks"]

            # Expand mask T if this batch has larger T than current
            if T_mask > dm.shape[1]:
                dm.resize((dm.shape[0], T_mask))

            # If the dataset has more columns than the current mask, right-pad the mask
            if dm.shape[1] > T_mask:
                pad_T = dm.shape[1] - T_mask
                m_np = np.pad(
                    m_np,
                    ((0, 0), (0, pad_T)),        # pad along T on the right
                    mode="constant",
                    constant_values=0,
                )
                # (optional) enforce exact shape
                if m_np.shape != (B, dm.shape[1]):
                    m_np = m_np.reshape(B, dm.shape[1])

            # Append rows
            dm.resize(self.count + B, axis=0)
            dm[self.count:self.count + B, :] = m_np


        # ---- Advance sample counter ----
        self.count += B


    # --- helpers to add inside the class ---

    # ensure dataset creation lets T/D grow later
    def _ensure_ds_shape(self, name: str, T_needed: int, D_needed: int):
        if name not in self.ds:
            self.ds[name] = self.f.create_dataset(
                name,
                shape=(0, max(1, T_needed), D_needed),
                maxshape=(None, None, None),   # allow future T/D growth
                chunks=(max(1, 4096 // max(1, T_needed * D_needed)), max(1, T_needed), D_needed),
                compression=self.compression,
                dtype=self.dtype,
            )
            return
        ds = self.ds[name]
        _, T_cur, D_cur = ds.shape
        new_T = max(T_cur, T_needed, 1)
        new_D = max(D_cur, D_needed)
        if new_T != T_cur or new_D != D_cur:
            ds.resize((ds.shape[0], new_T, new_D))

    def _mk_masks(self, T_init: int):
        if "masks" in self.f:
            return
        self.f.create_dataset(
            "masks",
            shape=(0, T_init),
            maxshape=(None, None),        # allow T to grow later
            chunks=(max(1, 4096 // max(1, T_init)), T_init),
            compression=self.compression,
            dtype=np.uint8,
        )

    # small local helper (outside the class is fine, or a @staticmethod)
    def _np_pad(arr, pad_spec, pad_value, target_shape=None):
        import numpy as _np
        out = _np.pad(arr, pad_spec, mode="constant", constant_values=pad_value)
        if target_shape is not None and tuple(out.shape) != tuple(target_shape):
            # final guard rail to exact-shape writes
            out = out.reshape(target_shape)
        return out




