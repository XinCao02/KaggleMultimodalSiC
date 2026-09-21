"""Compact residual predictor and a clearly scoped SS-OPM architecture adaptation.

SS-OPM structure/loss derived from Shuji Suzuki (2022), MIT licensed.
See THIRD_PARTY_NOTICES.md and docs/public_models.md for exact deviations.
"""

import numpy as np
import torch
from torch import nn
from .metrics import correlation_loss


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ResidualPredictor(nn.Module):
    def __init__(self, dim, outputs=140, width=512):
        super().__init__()
        self.linear = nn.Linear(dim, outputs)
        self.net = nn.Sequential(
            nn.Linear(dim, width),
            ResidualBlock(width),
            ResidualBlock(width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, outputs),
        )

    def forward(self, x):
        return self.linear(x) + self.net(x)

    def loss(self, x, y, progress=0):
        p = self(x)
        return correlation_loss(p, y) + 0.1 * nn.functional.mse_loss(p, y)

    def predict(self, x):
        return self(x)


class WinnerBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class SSOPMAdaptation(nn.Module):
    """Original 2048-wide one-encoder/five-decoder, six supervised output heads.

    Common RNA features replace original imputation/pathway/metadata pipeline.
    No gender embedding or annotation-derived batch statistics are used.
    """

    def __init__(self, dim, y_train, groups, width=2048):
        super().__init__()
        y = np.asarray(y_train, dtype="float32")
        lo, hi = np.quantile(y, [0.01, 0.99], axis=1).astype("float32")
        z = np.clip(y, lo[:, None], hi[:, None])
        median = np.median(z, axis=1, keepdims=True)
        z = z / np.where(np.abs(median) > 1e-6, median, 1.0)
        z = (z - z.mean(1, keepdims=True)) / np.maximum(z.std(1, keepdims=True), 1e-6)
        medians = {
            g: np.median(z[np.asarray(groups) == g], axis=0) for g in np.unique(groups)
        }
        global_median = np.mean(list(medians.values()), axis=0)
        residual = z - np.stack([medians[g] for g in groups])
        _, _, v = np.linalg.svd(residual, full_matrices=False)
        basis = v[: min(128, len(v))].astype("float32")
        target = residual @ basis.T
        for name, array in [
            ("basis", basis),
            ("offset", global_median),
            ("loc", target.mean(0)),
            ("scale", np.maximum(target.std(0), 1e-6)),
        ]:
            self.register_buffer(
                name, torch.from_numpy(np.asarray(array, dtype="float32"))
            )
        self.latent_targets = torch.from_numpy(target)
        self.in_fc = nn.Linear(dim, width)
        self.encoder = nn.Sequential(
            WinnerBlock(width, 0.5952997562668841), nn.Linear(width, width)
        )
        self.decoder_in = nn.Linear(width, width)
        self.blocks = nn.ModuleList(
            [WinnerBlock(width, 0.31846059114042935) for _ in range(5)]
        )
        self.heads = nn.ModuleList([nn.Linear(width, len(basis)) for _ in range(6)])

    def forward(self, x):
        h = self.decoder_in(self.encoder(self.in_fc(x)))
        hs = [h]
        for block in self.blocks:
            h = block(h)
            hs.append(h)
        return [head(h) * self.scale + self.loc for head, h in zip(self.heads, hs)]

    def loss(self, x, y, progress=0, latent=None):
        preds = self(x)
        return sum(
            correlation_loss(p @ self.basis + self.offset, y)
            + (1 - progress) ** 2 * nn.functional.l1_loss(p, latent)
            for p in preds
        ) / len(preds)

    def predict(self, x):
        ps = [p @ self.basis + self.offset for p in self(x)]
        ps = [
            (p - p.mean(1, keepdim=True))
            / p.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
            for p in ps
        ]
        return torch.stack(ps).mean(0)
