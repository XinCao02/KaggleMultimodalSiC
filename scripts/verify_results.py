"""Audit frozen results and verify end-to-end checkpoint reload on real test cells."""

import hashlib, json, pickle, subprocess, sys, tempfile
from pathlib import Path
import h5py, numpy as np, pandas as pd
from citebench.metrics import pearson_rows

root = Path(".")
d = root / "data/processed"
y = np.load(d / "y_train.npy")
meta = pd.read_csv(d / "meta_train.csv")
proteins = np.load(d / "proteins.npy")
checks = []
for s in ["temporal", "donor_13176", "donor_31800", "donor_32606", "random"]:
    out = root / "results" / s
    ix = np.load(out / "split_indices.npz")
    a, b, c = [ix[k] for k in ["train", "validation", "test"]]
    assert len(set(a) | set(b) | set(c)) == len(y)
    assert not (set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
    with open(out / "features.pkl", "rb") as f:
        features = pickle.load(f)
    assert np.array_equal(features.fit_indices, a)
    summary = pd.read_csv(out / "summary.csv")
    for row in summary.itertuples():
        p = np.load(out / row.model / "predictions.npy")
        assert p.shape == (len(c), len(proteins)) and np.isfinite(p).all()
        value = float(pearson_rows(y[c], p).mean())
        assert abs(value - row.cell_pearson) < 1e-10
        checks.append(
            {
                "protocol": s,
                "model": row.model,
                "cells": len(c),
                "metric_recomputed": value,
            }
        )
    for m in ["residual_s42", "residual_s43", "residual_s44", "ssopm_adapted"]:
        info = json.loads((out / m / "training.json").read_text())
        assert info["device"] == "mps"
        hist = pd.read_csv(out / m / "history.csv")
        best = hist.loc[hist.validation_pearson.idxmax()]
        assert int(best.epoch) == info["best_epoch"]
        assert abs(best.validation_pearson - info["validation_pearson"]) < 1e-12
final = root / "results/final"
f = np.load(final / "cite_predictions.npz")
mt = pd.read_csv(d / "meta_test.csv")
assert f["predictions"].shape == (48663, 140) and np.isfinite(f["predictions"]).all()
assert np.array_equal(f["cell_ids"], mt.cell_id.values)
assert np.array_equal(f["proteins"], proteins)
# Real original RNA profiles, held out from all model selection; never synthetic.
x = np.load(d / "x_test.npy", mmap_mode="r")
genes = np.load(d / "genes.npy")
n = 96
with tempfile.TemporaryDirectory(prefix="citebench_verify_") as tmp:
    tmp = Path(tmp)
    hfile = tmp / "real_test_subset.h5"
    predfile = tmp / "reload_predictions.npz"
    with h5py.File(hfile, "w") as h:
        g = h.create_group("test_cite_inputs")
        g.create_dataset("axis0", data=genes.astype("S"))
        g.create_dataset("axis1", data=f["cell_ids"][:n].astype("S"))
        g.create_dataset("block0_values", data=x[:n])
    subprocess.run(
        [
            sys.executable,
            "scripts/predict.py",
            "--model-dir",
            str(final),
            "--input",
            str(hfile),
            "--output",
            str(predfile),
            "--device",
            "mps",
        ],
        check=True,
    )
    reloaded = np.load(predfile)
    error = float(np.max(np.abs(reloaded["predictions"] - f["predictions"][:n])))
    assert np.array_equal(reloaded["cell_ids"], f["cell_ids"][:n])
    assert error < 2e-5, error
result = {
    "status": "passed",
    "frozen_model_protocol_results": len(checks),
    "metric_recomputation_max_tolerance": 1e-10,
    "checkpoint_selection_verified": True,
    "all_neural_training_device": "mps",
    "final_prediction_shape": list(f["predictions"].shape),
    "real_cells_in_reload_test": n,
    "reload_max_absolute_error": error,
    "reload_tolerance": 2e-5,
    "checks": checks,
}
(root / "results/verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
manifest = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ["src", "scripts", "tests"]
    for p in Path(folder).rglob("*.py")
}
(root / "results/source_manifest.json").write_text(json.dumps(manifest, indent=2))
