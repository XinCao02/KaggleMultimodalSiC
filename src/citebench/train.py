import copy
import gc
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .metrics import (
    pearson_rows,
    protein_correlations,
    paired_cluster_bootstrap,
    row_normalize,
)
from .models import ResidualPredictor, SSOPMAdaptation


def require_device(name="mps"):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS required but unavailable. On macOS run outside a GPU-blocking sandbox; no silent CPU fallback."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    if name == "mps":
        torch.mps.set_per_process_memory_fraction(0.60)
    return name


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def predict(model, x, device, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(x), batch):
            out.append(
                model.predict(torch.as_tensor(x[j : j + batch], device=device))
                .cpu()
                .numpy()
            )
    return np.vstack(out)


def fit_nn(
    x,
    y,
    xval,
    yval,
    groups,
    out,
    kind="residual",
    seed=42,
    epochs=40,
    device="mps",
    width=2048,
    fixed_epochs=None,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    t0 = time.monotonic()
    # Unit variance per cell is aligned with Pearson; preserves all 140 targets.
    yn = (y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6)
    model = (
        ResidualPredictor(x.shape[1], y.shape[1])
        if kind == "residual"
        else SSOPMAdaptation(x.shape[1], y, groups, width=width)
    )
    model = model.to(device)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(yn, dtype=torch.float32, device=device)
    latent = model.latent_targets.to(device) if kind == "ssopm" else None
    batch = 256
    lr = 1e-3 if kind == "residual" else 0.00012520653814999459
    wd = 1e-4 if kind == "residual" else 2.576638574613591e-6
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if kind == "residual"
        else torch.optim.Adam(
            model.parameters(), lr=lr, eps=7.257005721594269e-8, weight_decay=wd
        )
    )
    nepochs = fixed_epochs or epochs
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)
        if kind == "residual"
        else torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=nepochs * ((len(x) + batch - 1) // batch),
            pct_start=0.3,
        )
    )
    best = -float("inf")
    best_state = None
    best_epoch = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(nepochs):
        model.train()
        order = rng.permutation(len(x))
        ls = 0.0
        progress = max(0.0, (epoch - 10) / max(1, nepochs - 10))
        for j in range(0, len(x), batch):
            ix = torch.as_tensor(order[j : j + batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "ssopm":
                loss = model.loss(xt[ix], yt[ix], progress, latent[ix])
            else:
                loss = model.loss(xt[ix], yt[ix], progress)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if kind == "ssopm":
                scheduler.step()
            ls += float(loss.detach().cpu()) * len(ix)
        if kind == "residual":
            scheduler.step()
        score = (
            float(pearson_rows(yval, predict(model, xval, device)).mean())
            if xval is not None
            else None
        )
        if fixed_epochs or score > best:
            if score is not None:
                best = score
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        entry = dict(
            epoch=epoch + 1,
            loss=ls / len(x),
            validation_pearson=score,
            seconds=time.monotonic() - t0,
        )
        history.append(entry)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(kind, seed, entry, flush=True)
        if not fixed_epochs and epoch + 1 - best_epoch >= 10 and epoch >= 19:
            break
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "kind": kind,
            "input_dim": x.shape[1],
            "width": width,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        out / "model.pt",
    )
    info = {
        "kind": kind,
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "validation_pearson": best if xval is not None else None,
        "seconds": time.monotonic() - t0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "batch_size": batch,
        "learning_rate": lr,
        "mps_allocated_bytes": (
            int(torch.mps.current_allocated_memory()) if device == "mps" else None
        ),
        "mps_driver_bytes": (
            int(torch.mps.driver_allocated_memory()) if device == "mps" else None
        ),
    }
    (out / "training.json").write_text(json.dumps(info, indent=2))
    del xt, yt, latent
    gc.collect()
    return model, info


def evaluate(y, p, meta, proteins, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cell = pearson_rows(y, p)
    frame = meta.copy()
    frame["pearson"] = cell
    frame.to_csv(out / "cell_scores.csv", index=False)
    pc = protein_correlations(y, p)
    pd.DataFrame(
        {"protein": proteins, "pearson_across_cells": pc, "target_std": y.std(0)}
    ).to_csv(out / "protein_scores.csv", index=False)
    summary = {
        "cell_pearson": float(cell.mean()),
        "median_cell_pearson": float(np.median(cell)),
        "mean_protein_pearson": (
            float(np.nanmean(pc)) if np.isfinite(pc).any() else None
        ),
        "cells": len(y),
        "proteins": y.shape[1],
        "donor_day_macro_pearson": float(frame.groupby("batch").pearson.mean().mean()),
    }
    for column in ["donor", "day", "cell_type", "batch"]:
        frame.groupby(column).pearson.agg(
            ["mean", "median", "size"]
        ).reset_index().to_csv(out / f"by_{column}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.save(out / "predictions.npy", p)
    return summary
