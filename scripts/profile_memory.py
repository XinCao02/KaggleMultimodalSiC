"""Measure MPS allocations in isolated training-step probes using real temporal data.

Reports the maximum synchronized stage sample, not a hardware peak counter.
Each model/trial runs in a fresh process; frozen checkpoints are never modified.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def probe(kind, trial, output):
    import numpy as np
    import pandas as pd
    import torch
    from citebench.models import ResidualPredictor, SSOPMAdaptation
    from citebench.train import require_device, seed_all

    require_device("mps")
    seed_all(42)
    tr = np.load("results/temporal/split_indices.npz")["train"]
    x = np.load("results/temporal/features_train.npy")
    y = np.load("data/processed/y_train.npy")[tr]
    groups = pd.read_csv("data/processed/meta_train.csv").iloc[tr].batch.values
    samples = []

    def sample(stage, step=-1):
        torch.mps.synchronize()
        samples.append(
            dict(
                stage=stage,
                step=step,
                tensor_bytes=int(torch.mps.current_allocated_memory()),
                driver_bytes=int(torch.mps.driver_allocated_memory()),
            )
        )

    sample("baseline")
    model = (
        ResidualPredictor(x.shape[1])
        if kind == "residual"
        else SSOPMAdaptation(x.shape[1], y, groups)
    ).to("mps")
    xt = torch.as_tensor(x, device="mps")
    yn = (y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6)
    yt = torch.as_tensor(yn, dtype=torch.float32, device="mps")
    latent = model.latent_targets.to("mps") if kind == "ssopm" else None
    opt = (
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if kind == "residual"
        else torch.optim.Adam(
            model.parameters(),
            lr=0.00012520653814999459,
            eps=7.257005721594269e-8,
            weight_decay=2.576638574613591e-6,
        )
    )
    sample("model_and_training_data")
    model.train()
    for step in range(10):
        ix = torch.arange(step * 256, (step + 1) * 256, device="mps")
        opt.zero_grad(set_to_none=True)
        loss = (
            model.loss(xt[ix], yt[ix], 0, latent[ix])
            if kind == "ssopm"
            else model.loss(xt[ix], yt[ix])
        )
        sample("forward", step)
        loss.backward()
        sample("backward", step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sample("optimizer", step)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(xt[:1024])
        sample("evaluation_batch")
        assert torch.isfinite(prediction).all()
    result = dict(
        model=kind,
        trial=trial,
        device="mps",
        dtype="float32",
        torch_version=torch.__version__,
        batch_size=256,
        evaluation_batch_size=1024,
        train_cells=len(tr),
        feature_dim=x.shape[1],
        optimizer="AdamW" if kind == "residual" else "Adam",
        measurement="Maximum synchronized stage sample; not hardware peak",
        tensor_max_bytes=max(s["tensor_bytes"] for s in samples),
        driver_max_bytes=max(s["driver_bytes"] for s in samples),
        samples=samples,
    )
    output.write_text(json.dumps(result, indent=2))
    print(kind, trial, result["tensor_max_bytes"] / 2**20, "MiB", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["residual", "ssopm"])
    p.add_argument("--trial", type=int, default=0)
    args = p.parse_args()
    out = Path("results/memory")
    out.mkdir(parents=True, exist_ok=True)
    if args.model:
        probe(args.model, args.trial, out / f"{args.model}_{args.trial}.json")
        return
    for kind in ["residual", "ssopm"]:
        for trial in range(3):
            subprocess.run(
                [sys.executable, __file__, "--model", kind, "--trial", str(trial)],
                check=True,
            )
    import pandas as pd

    rows = [json.loads(f.read_text()) for f in sorted(out.glob("*_[0-2].json"))]
    pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in rows]).to_csv(
        out / "summary.csv", index=False
    )


if __name__ == "__main__":
    main()
