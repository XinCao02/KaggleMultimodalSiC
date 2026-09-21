"""Predeclared temporal feature ablations; seed 42, same checkpoint rule."""

import json
from pathlib import Path
import numpy as np, pandas as pd
from citebench.train import fit_nn, predict, evaluate, require_device

if __name__ == "__main__":
    device = require_device("mps")
    root = Path("results/temporal")
    data = Path("data/processed")
    ix = np.load(root / "split_indices.npz")
    y = np.load(data / "y_train.npy")
    meta = pd.read_csv(data / "meta_train.csv")
    proteins = np.load(data / "proteins.npy")
    xs = [np.load(root / f"features_{p}.npy") for p in ["train", "validation", "test"]]
    result = []
    for name, cols in [
        ("pca_only", np.arange(128)),
        ("selected_genes_only", np.arange(128, xs[0].shape[1] - 4)),
    ]:
        dest = root / "ablations" / name
        m, info = fit_nn(
            xs[0][:, cols].copy(),
            y[ix["train"]],
            xs[1][:, cols].copy(),
            y[ix["validation"]],
            meta.iloc[ix["train"]].batch.values,
            dest,
            seed=42,
            device=device,
        )
        pred = predict(m, xs[2][:, cols].copy(), device)
        score = evaluate(
            y[ix["test"]],
            pred,
            meta.iloc[ix["test"]].reset_index(drop=True),
            proteins,
            dest,
        )
        result.append(dict(model=name, **score, **info))
    pd.DataFrame(result).to_csv(root / "ablations/summary.csv", index=False)
