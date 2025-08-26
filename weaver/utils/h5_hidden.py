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

    def append(self, layer_tensors, mask=None, layer_names=None):
        """
        Accepts hidden states as:
        - dict[str, Tensor]                 # names are the dict keys
        - list/tuple[Tensor]                # names auto: hidden_1..N (or use layer_names)
        - Tensor of shape (L, B, T, D)      # split along L
        - Tensor of shape (B, T, D)         # single layer
        Optionally accepts a mask of shape (B, T) (bool/uint8).
        """

        # ---- Normalize inputs into (names, tensors) without external helpers ----
        names = None
        tensors = None

        if isinstance(layer_tensors, dict):
            # Keep provided order
            names = list(layer_tensors.keys())
            tensors = [layer_tensors[k] for k in names]

        elif isinstance(layer_tensors, (list, tuple)):
            tensors = list(layer_tensors)
            names = layer_names or [f"hidden_{i+1}" for i in range(len(tensors))]

        elif isinstance(layer_tensors, torch.Tensor):
            if layer_tensors.ndim == 4:
                # (L, B, T, D) -> split along L
                L = layer_tensors.shape[0]
                tensors = list(layer_tensors.unbind(0))
                names = layer_names or [f"hidden_{i+1}" for i in range(L)]
            elif layer_tensors.ndim == 3:
                tensors = [layer_tensors]
                names = layer_names or ["hidden_1"]
            else:
                raise ValueError(f"Unsupported tensor rank: {layer_tensors.ndim}, expected 3 or 4.")
        else:
            raise TypeError("layer_tensors must be dict, sequence of Tensors, or a Tensor.")

        if len(tensors) == 0:
            return
        if len(names) != len(tensors):
            raise ValueError(f"layer_names length ({len(names)}) must match number of tensors ({len(tensors)}).")

        # ---- Helper to standardize to (B, T, D) on CPU ----
        def to_btd_cpu(x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(f"Each layer must be rank-3 (B,T,D or T,B,D); got {tuple(x.shape)}")
            # Accept (B,T,D) or (T,B,D)
            if x.shape[0] < x.shape[1]:   # likely (T,B,D)
                x = x.permute(1, 0, 2)
            return x.contiguous().to("cpu")

        # ---- Determine (B, T, D) from the first tensor, create datasets lazily ----
        first = to_btd_cpu(tensors[0])
        B, T, D = first.shape

        for n in names:
            if n not in self.ds:
                self._mk(n, T, D)

        # ---- Write each layer ----
        for n, t in zip(names, tensors):
            x = to_btd_cpu(t)
            if tuple(x.shape) != (B, T, D):
                raise ValueError(f"Inconsistent shapes in batch: got {tuple(x.shape)} vs expected {(B, T, D)}")
            ds = self.ds[n]
            ds.resize(self.count + B, axis=0)
            ds[self.count:self.count + B, :, :] = x.numpy().astype(self.dtype, copy=False)

        # ---- Optional mask ----
        if mask is not None:
            m = mask
            if isinstance(m, torch.Tensor):
                m = m.to("cpu").to(torch.uint8).numpy()
            elif isinstance(m, np.ndarray):
                m = m.astype(np.uint8, copy=False)
            else:
                raise TypeError("mask must be a Tensor or numpy array.")

            if m.shape != (B, T):
                raise ValueError(f"Mask shape {m.shape} does not match (B, T) = {(B, T)}")
            self._mk_masks(T)
            dm = self.f["masks"]
            dm.resize(self.count + B, axis=0)
            dm[self.count:self.count + B, :] = m

        self.count += B


    def close(self):
        self.f.flush(); self.f.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

