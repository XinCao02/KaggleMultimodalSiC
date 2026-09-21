"""Training-only feature fitting with GPU matrix products and bounded host memory."""

import gc
import numpy as np
import torch


def gpu_mm(a, b, device):
    with torch.no_grad():
        c = torch.as_tensor(a, dtype=torch.float32, device=device) @ torch.as_tensor(
            b, dtype=torch.float32, device=device
        )
        return c.cpu().numpy()


def randomized_components(a, k=128, seed=2026, device="mps", iterations=3):
    """MPS matmul; CPU QR/SVD because MPS does not implement all LAPACK ops."""
    k = min(k, min(a.shape) - 1)
    rng = np.random.default_rng(seed)
    q = min(k + 20, min(a.shape))
    omega = rng.standard_normal((a.shape[1], q)).astype("float32")
    with torch.no_grad():
        at = torch.as_tensor(
            np.ascontiguousarray(a), dtype=torch.float32, device=device
        )
        z = (at @ torch.as_tensor(omega, device=device)).cpu().numpy()
        for _ in range(iterations):
            qleft = np.linalg.qr(z, mode="reduced")[0].astype("float32")
            zright = (at.T @ torch.as_tensor(qleft, device=device)).cpu().numpy()
            qright = np.linalg.qr(zright, mode="reduced")[0].astype("float32")
            z = (at @ torch.as_tensor(qright, device=device)).cpu().numpy()
        qleft = np.linalg.qr(z, mode="reduced")[0].astype("float32")
        b = (torch.as_tensor(qleft.T.copy(), device=device) @ at).cpu().numpy()
        _, s, v = np.linalg.svd(b, full_matrices=False)
        del at
    return v[:k].astype("float32"), s[:k]


class RNAFeatures:
    def __init__(
        self, n_hvg=4000, n_components=128, top_per_target=3, seed=2026, device="mps"
    ):
        self.n_hvg = n_hvg
        self.n_components = n_components
        self.top_per_target = top_per_target
        self.seed = seed
        self.device = device

    def fit(self, x, y, indices, meta, genes):
        self.fit_indices = np.asarray(indices).copy()
        n = len(indices)
        s = np.zeros(x.shape[1])
        ss = s.copy()
        for j in range(0, n, 1024):
            a = np.asarray(x[indices[j : j + 1024]], dtype="float32")
            s += a.sum(0, dtype=np.float64)
            ss += np.einsum("ij,ij->j", a, a, dtype=np.float64)
        var = np.maximum(ss / n - (s / n) ** 2, 0)
        self.hvg = np.argsort(var)[-min(self.n_hvg, (var > 1e-8).sum()) :]
        a = np.empty((n, len(self.hvg)), dtype="float32")
        for j in range(0, n, 1024):
            a[j : j + 1024] = x[indices[j : j + 1024]][:, self.hvg]
        self.pca_mean = a.mean(0)
        centered = a - self.pca_mean
        print("Fitting RNA components on", a.shape, "device", self.device, flush=True)
        self.components, self.singular_values = randomized_components(
            centered, self.n_components, self.seed, self.device
        )
        corrs = []
        for group in sorted(meta.iloc[indices].batch.unique()):
            mask = meta.iloc[indices].batch.values == group
            ag = a[mask]
            yg = np.asarray(y[indices[mask]], dtype="float32")
            ag = (ag - ag.mean(0)) / np.maximum(ag.std(0), 1e-5)
            yg = (yg - yg.mean(0)) / np.maximum(yg.std(0), 1e-5)
            corrs.append(gpu_mm(ag.T, yg, self.device) / len(ag))
        stable = np.abs(np.median(np.stack(corrs), axis=0))
        selected = np.unique(np.argsort(stable, axis=0)[-self.top_per_target :].ravel())
        self.selected = self.hvg[selected]
        # Directly named lineage markers: fixed before fitting, RNA features only.
        markers = {
            "CD34",
            "CD38",
            "KIT",
            "GATA1",
            "GATA2",
            "SPI1",
            "MPO",
            "ELANE",
            "MS4A1",
            "CD79A",
            "ITGA2B",
            "PF4",
            "HBB",
            "HBA1",
            "LYZ",
            "FCER1A",
            "IL3RA",
            "PTPRC",
        }
        symbols = np.array(
            [
                (
                    g.split("_", 1)[1]
                    if g.startswith("ENSG") and "_" in g
                    else g.split("_")[0]
                )
                for g in genes
            ]
        )
        self.selected = np.union1d(
            self.selected, np.flatnonzero(np.isin(symbols, list(markers)))
        )
        self.selected_names = genes[self.selected]
        raw = self._raw(x, indices)
        self.mean = raw.mean(0)
        self.scale = np.maximum(raw.std(0), 1e-4)
        self.output_dim = raw.shape[1]
        del a, centered, raw
        gc.collect()
        return self

    def _raw(self, x, indices):
        rows = []
        for j in range(0, len(indices), 1024):
            a = np.asarray(x[indices[j : j + 1024]], dtype="float32")
            pcs = gpu_mm(a[:, self.hvg] - self.pca_mean, self.components.T, self.device)
            qc = np.column_stack(
                [(a > 0).mean(1), a.mean(1), a.std(1), np.log1p(np.expm1(a).sum(1))]
            )
            rows.append(
                np.column_stack([pcs, a[:, self.selected], qc]).astype("float32")
            )
        return np.vstack(rows)

    def transform(self, x, indices):
        return np.clip(
            (self._raw(x, indices) - self.mean) / self.scale, -10, 10
        ).astype("float32")
