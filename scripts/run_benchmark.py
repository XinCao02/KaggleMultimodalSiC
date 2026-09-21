"""Nested holdouts: choose hyperparameters/checkpoints only on the validation set."""

import argparse
import gc
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from citebench.data import make_splits
from citebench.features import RNAFeatures
from citebench.metrics import pearson_rows, row_normalize, paired_cluster_bootstrap
from citebench.train import require_device, fit_nn, predict, evaluate


def run(args):
    root = Path(args.root)
    data = root / "data/processed"
    device = require_device(args.device)
    x = np.load(data / "x_train.npy", mmap_mode="r")
    y = np.load(data / "y_train.npy")
    meta = pd.read_csv(data / "meta_train.csv")
    genes = np.load(data / "genes.npy")
    proteins = np.load(data / "proteins.npy")
    splits = make_splits(meta)
    for name in args.splits:
        out = root / "results" / name
        out.mkdir(parents=True, exist_ok=True)
        tr, va, te = splits[name]
        np.savez(out / "split_indices.npz", train=tr, validation=va, test=te)
        meta.assign(
            split=np.select(
                [np.isin(np.arange(len(meta)), tr), np.isin(np.arange(len(meta)), va)],
                ["train", "validation"],
                default="test",
            )
        ).to_csv(out / "split_cells.csv", index=False)
        print("START", name, len(tr), len(va), len(te), flush=True)
        if (out / "features.pkl").exists():
            with open(out / "features.pkl", "rb") as f:
                fe = pickle.load(f)
            assert np.array_equal(fe.fit_indices, tr)
            fe.device = device
        else:
            t = time.monotonic()
            fe = RNAFeatures(device=device).fit(x, y, tr, meta, genes)
            with open(out / "features.pkl", "wb") as f:
                pickle.dump(fe, f)
            (out / "features.json").write_text(
                json.dumps(
                    {
                        "fit_cells": len(tr),
                        "hvg": len(fe.hvg),
                        "pc": len(fe.components),
                        "supervised_and_marker_genes": len(fe.selected),
                        "dim": fe.output_dim,
                        "seconds": time.monotonic() - t,
                    },
                    indent=2,
                )
            )
            pd.DataFrame({"gene": fe.selected_names}).to_csv(
                out / "selected_genes.csv", index=False
            )
        arrays = []
        for part, ix in [("train", tr), ("validation", va), ("test", te)]:
            dest = out / f"features_{part}.npy"
            if not dest.exists():
                np.save(dest, fe.transform(x, ix))
            arrays.append(np.load(dest))
        xt, xv, xe = arrays
        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "split": name,
                    "train": len(tr),
                    "validation": len(va),
                    "test": len(te),
                    "selection": "Validation only; test never used for model selection",
                    "seeds": args.seeds,
                    "max_epochs": args.epochs,
                    "ssopm_width": args.winner_width,
                },
                indent=2,
            )
        )
        summaries = []

        def record(label, p, extra=None):
            s = evaluate(
                y[te], p, meta.iloc[te].reset_index(drop=True), proteins, out / label
            )
            summaries.append(dict(model=label, **s, **(extra or {})))
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)

        mean_pred = np.tile(y[tr].mean(0), (len(te), 1))
        record("mean_profile", mean_pred)
        best = (-np.inf, None, None)
        grid = []
        yn = row_normalize(y[tr]) * np.sqrt(y.shape[1])
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            model = Ridge(alpha=alpha).fit(xt, yn)
            score = pearson_rows(y[va], model.predict(xv)).mean()
            grid.append({"alpha": alpha, "validation_pearson": score})
            if score > best[0]:
                best = (score, model, alpha)
        pd.DataFrame(grid).to_csv(out / "ridge_tuning.csv", index=False)
        ridge = best[1]
        record("ridge", ridge.predict(xe), {"alpha": best[2]})
        with open(out / "ridge.pkl", "wb") as f:
            pickle.dump(ridge, f)
        preds = []
        vpreds = []
        for seed in args.seeds:
            label = f"residual_s{seed}"
            if (out / label / "predictions.npy").exists() and (
                out / label / "validation.npy"
            ).exists():
                p = np.load(out / label / "predictions.npy")
                v = np.load(out / label / "validation.npy")
            else:
                model, info = fit_nn(
                    xt,
                    y[tr],
                    xv,
                    y[va],
                    meta.iloc[tr].batch.values,
                    out / label,
                    seed=seed,
                    epochs=args.epochs,
                    device=device,
                )
                p = predict(model, xe, device)
                v = predict(model, xv, device)
                np.save(out / label / "validation.npy", v)
                del model
                gc.collect()
            record(label, p)
            preds.append(row_normalize(p))
            vpreds.append(row_normalize(v))
        ens = np.mean(preds, axis=0)
        vens = np.mean(vpreds, axis=0)
        record("residual_ensemble", ens)
        rv = row_normalize(ridge.predict(xv))
        re = row_normalize(ridge.predict(xe))
        grid = [
            {
                "neural_weight": float(w),
                "validation_pearson": float(
                    pearson_rows(y[va], w * vens + (1 - w) * rv).mean()
                ),
            }
            for w in np.linspace(0, 1, 11)
        ]
        w = max(grid, key=lambda z: z["validation_pearson"])["neural_weight"]
        pd.DataFrame(grid).to_csv(out / "blend_tuning.csv", index=False)
        blend = w * ens + (1 - w) * re
        record("selected_blend", blend, {"neural_weight": w})
        if not args.skip_winner:
            label = "ssopm_adapted"
            if (out / label / "predictions.npy").exists():
                p = np.load(out / label / "predictions.npy")
            else:
                model, info = fit_nn(
                    xt,
                    y[tr],
                    xv,
                    y[va],
                    meta.iloc[tr].batch.values,
                    out / label,
                    kind="ssopm",
                    seed=42,
                    epochs=args.epochs,
                    device=device,
                    width=args.winner_width,
                )
                p = predict(model, xe, device)
                np.save(out / label / "validation.npy", predict(model, xv, device))
                del model
                gc.collect()
            record(label, p)
            ci = paired_cluster_bootstrap(y[te], blend, p, meta.iloc[te].batch.values)
            (out / "paired_comparison.json").write_text(json.dumps(ci, indent=2))
        print(
            "COMPLETE",
            name,
            pd.DataFrame(summaries)[["model", "cell_pearson"]].to_string(index=False),
            flush=True,
        )
        del arrays, xt, xv, xe
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument(
        "--splits",
        nargs="+",
        default=["temporal", "donor_13176", "donor_31800", "donor_32606", "random"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--winner-width", type=int, default=2048)
    p.add_argument("--skip-winner", action="store_true")
    run(p.parse_args())
