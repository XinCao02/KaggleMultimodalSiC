"""Refit the validation-selected compact pipeline and export all original test cells."""

import argparse
import gc
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from citebench.features import RNAFeatures
from citebench.metrics import row_normalize
from citebench.train import require_device, fit_nn, predict


def main(args):
    root = Path(args.root)
    data = root / "data/processed"
    out = root / "results/final"
    out.mkdir(parents=True, exist_ok=True)
    device = require_device(args.device)
    reference = root / "results/temporal"
    # Selection is based only on day-3 validation, never day-4 test performance.
    epochs = [
        json.loads((reference / f"residual_s{s}/training.json").read_text())[
            "best_epoch"
        ]
        for s in [42, 43, 44]
    ]
    epoch = int(np.median(epochs))
    grid = pd.read_csv(reference / "blend_tuning.csv")
    weight = float(grid.loc[grid.validation_pearson.idxmax(), "neural_weight"])
    grid = pd.read_csv(reference / "ridge_tuning.csv")
    alpha = float(grid.loc[grid.validation_pearson.idxmax(), "alpha"])
    x = np.load(data / "x_train.npy", mmap_mode="r")
    test = np.load(data / "x_test.npy", mmap_mode="r")
    y = np.load(data / "y_train.npy")
    meta = pd.read_csv(data / "meta_train.csv")
    mt = pd.read_csv(data / "meta_test.csv")
    genes = np.load(data / "genes.npy")
    proteins = np.load(data / "proteins.npy")
    indices = np.arange(len(x))
    it = np.arange(len(test))
    if (out / "features.pkl").exists():
        with open(out / "features.pkl", "rb") as f:
            fe = pickle.load(f)
        fe.device = device
    else:
        fe = RNAFeatures(device=device).fit(x, y, indices, meta, genes)
        with open(out / "features.pkl", "wb") as f:
            pickle.dump(fe, f)
    xt = fe.transform(x, indices)
    xe = fe.transform(test, it)
    np.save(out / "feature_test.npy", xe)
    ridge = Ridge(alpha=alpha).fit(xt, row_normalize(y) * np.sqrt(y.shape[1]))
    with open(out / "ridge.pkl", "wb") as f:
        pickle.dump(ridge, f)
    ps = []
    for seed in [42, 43, 44]:
        dest = out / f"residual_s{seed}"
        model, info = fit_nn(
            xt,
            y,
            None,
            None,
            meta.batch.values,
            dest,
            seed=seed,
            device=device,
            fixed_epochs=epoch,
        )
        ps.append(row_normalize(predict(model, xe, device)))
        del model
        gc.collect()
    ensemble = np.mean(ps, axis=0)
    pred = weight * ensemble + (1 - weight) * row_normalize(ridge.predict(xe))
    np.savez_compressed(
        out / "cite_predictions.npz",
        predictions=pred,
        cell_ids=mt.cell_id.values.astype(str),
        proteins=proteins,
    )
    pd.DataFrame(
        pred, index=pd.Index(mt.cell_id, name="cell_id"), columns=proteins
    ).to_csv(out / "cite_predictions_wide.csv.gz", compression="gzip")
    config = {
        "training_cells": len(x),
        "test_cells": len(test),
        "proteins": len(proteins),
        "device": device,
        "epochs": epoch,
        "seeds": [42, 43, 44],
        "ridge_alpha": alpha,
        "neural_weight": weight,
        "selection_source": "Temporal split validation (day 3) only",
        "score_status": "Official test labels not used; no Kaggle score claimed",
        "prediction_scale": "Per-cell centered, unit-norm model blend; not absolute dsb units",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    np.save(out / "genes.npy", genes)
    np.save(out / "proteins.npy", proteins)
    print("Final predictions ready", config, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--device", default="mps")
    main(p.parse_args())
