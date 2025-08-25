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
        layer_tensors: List[Tensor] per layer, each (B,T,D) or (T,B,D)
        mask: Optional Tensor (B,T) (bool/uint8)
        layer_names: Optional List[str] same length as layer_tensors
        """
        # --- normalize to a list of tensors ---
        if isinstance(layer_tensors, torch.Tensor):
            layer_tensors = [layer_tensors]
        elif isinstance(layer_tensors, np.ndarray):
            # (B,T,D) single layer from numpy
            layer_tensors = [torch.from_numpy(layer_tensors)]
        elif isinstance(layer_tensors, (list, tuple)):
            layer_tensors = list(layer_tensors)
        else:
            raise TypeError("layer_tensors must be a Tensor or a sequence of Tensors")

        if len(layer_tensors) == 0:
            return
        
        x0 = self._to_btd(layer_tensors[0])
        B, T, D = x0.shape
        names = layer_names or [f"layer_{i}" for i in range(len(layer_tensors))]

        for n in names: self._mk(n, T, D)

        for n, t in zip(names, layer_tensors):
            x = self._to_btd(t)
            if tuple(x.shape) != (B, T, D):
                raise ValueError(f"shape mismatch {tuple(x.shape)} vs {(B,T,D)}")
            ds = self.ds[n]
            ds.resize(self.count + B, axis=0)
            ds[self.count:self.count+B, :, :] = x.numpy().astype(self.dtype, copy=False)

        if mask is not None:
            m = mask
            if isinstance(m, torch.Tensor):
                m = m.to("cpu").to(torch.uint8).numpy()
            if m.shape != (B, T):
                raise ValueError(f"mask shape {m.shape} != {(B,T)}")
            self._mk_masks(T)
            dm = self.f["masks"]
            dm.resize(self.count + B, axis=0)
            dm[self.count:self.count+B, :] = m

        self.count += B

    def close(self):
        self.f.flush(); self.f.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

