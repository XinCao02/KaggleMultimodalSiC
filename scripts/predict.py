"""Reload trusted local artifacts and predict a gene-aligned Kaggle-format HDF5."""

import argparse, json, pickle
from pathlib import Path
import hdf5plugin, h5py, numpy as np, pandas as pd, torch
from citebench.models import ResidualPredictor
from citebench.metrics import row_normalize
from citebench.train import require_device, predict

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="results/final")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    d = Path(a.model_dir)
    device = require_device(a.device)
    cfg = json.loads((d / "config.json").read_text())
    with open(d / "features.pkl", "rb") as f:
        fe = pickle.load(f)
    with open(d / "ridge.pkl", "rb") as f:
        ridge = pickle.load(f)
    fe.device = device
    expected = np.load(d / "genes.npy")
    proteins = np.load(d / "proteins.npy")
    feature_blocks = []
    ids = []
    with h5py.File(a.input) as h:
        g = h[next(iter(h))]
        genes = g["axis0"][:].astype(str)
        if not np.array_equal(genes, expected):
            raise ValueError("RNA gene identifiers/order differ from fitted pipeline")
        ids = g["axis1"][:].astype(str)
        ds = g["block0_values"]
        for j in range(0, len(ids), 1024):
            block = ds[j : j + 1024]
            feature_blocks.append(fe.transform(block, np.arange(len(block))))
    x = np.vstack(feature_blocks)
    ps = []
    for seed in cfg["seeds"]:
        checkpoint = torch.load(
            d / f"residual_s{seed}/model.pt", map_location="cpu", weights_only=True
        )
        model = ResidualPredictor(checkpoint["input_dim"], len(proteins)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        ps.append(row_normalize(predict(model, x, device)))
    pred = cfg["neural_weight"] * np.mean(ps, 0) + (
        1 - cfg["neural_weight"]
    ) * row_normalize(ridge.predict(x))
    np.savez_compressed(a.output, predictions=pred, cell_ids=ids, proteins=proteins)
    print("Predicted", pred.shape, "on", device)
