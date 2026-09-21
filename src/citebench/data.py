"""HDF5 streaming and identifier-first alignment; no full dense HDF load."""

from pathlib import Path
import hashlib
import json
import hdf5plugin  # registers the Blosc filter used by Kaggle
import h5py
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def read_h5(path):
    with h5py.File(path) as h:
        g = h[next(iter(h))]
        return (
            g["block0_values"][:],
            g["axis1"][:].astype(str),
            g["axis0"][:].astype(str),
        )


def prepare(data_dir, out_dir):
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y, ids_y, proteins = read_h5(data_dir / "train_cite_targets.h5")
    meta = pd.read_csv(data_dir / "metadata.csv").set_index(
        "cell_id", verify_integrity=True
    )
    manifest = {
        "files": {},
        "source": "Original Kaggle 2022 HDF5 files",
        "target_normalization": "dsb",
    }
    for split in ["train", "test"]:
        name = f"{split}_cite_inputs.h5"
        with h5py.File(data_dir / name) as h:
            g = h[f"{split}_cite_inputs"]
            values = g["block0_values"]
            ids = g["axis1"][:].astype(str)
            genes = g["axis0"][:].astype(str)
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate cell IDs")
            if split == "train":
                if not np.array_equal(ids, ids_y):
                    raise ValueError("RNA/ADT cell order mismatch")
                np.save(out / "y_train.npy", y)
                np.save(out / "genes.npy", genes)
                np.save(out / "proteins.npy", proteins)
            elif not np.array_equal(genes, np.load(out / "genes.npy")):
                raise ValueError("Test gene order mismatch")
            dest = out / f"x_{split}.npy"
            if not dest.exists():
                tmp = out / f"x_{split}.partial.npy"
                x = np.lib.format.open_memmap(
                    tmp, mode="w+", dtype="float32", shape=values.shape
                )
                for start in range(0, len(ids), 1024):
                    block = values[start : start + 1024]
                    if not np.isfinite(block).all() or (block < 0).any():
                        raise ValueError("Invalid RNA values")
                    x[start : start + len(block)] = block
                    if start % 16384 == 0:
                        print("Preparing", split, start, len(ids), flush=True)
                x.flush()
                del x
                tmp.rename(dest)
            m = meta.loc[ids].copy()
            if m.isna().any().any():
                raise ValueError("Missing metadata")
            m["batch"] = m.donor.astype(str) + "_d" + m.day.astype(str)
            m.to_csv(out / f"meta_{split}.csv")
            manifest[split] = {
                "cells": len(ids),
                "genes": len(genes),
                "donor_day_counts": m.groupby(["donor", "day"]).size().to_dict(),
            }
            manifest[split]["donor_day_counts"] = {
                str(k): v for k, v in manifest[split]["donor_day_counts"].items()
            }
        manifest["files"][name] = {
            "bytes": (data_dir / name).stat().st_size,
            "sha256": sha256(data_dir / name),
        }
    manifest["files"]["train_cite_targets.h5"] = {
        "sha256": sha256(data_dir / "train_cite_targets.h5")
    }
    manifest["proteins"] = len(proteins)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared and hashed original data", flush=True)


def make_splits(meta, seed=2026):
    from sklearn.model_selection import train_test_split

    ids = np.arange(len(meta))
    splits = {
        "temporal": (
            ids[meta.day.values == 2],
            ids[meta.day.values == 3],
            ids[meta.day.values == 4],
        )
    }
    donors = sorted(meta.donor.unique())
    for i, d in enumerate(donors):
        val = donors[(i + 1) % len(donors)]
        tr = donors[(i + 2) % len(donors)]
        splits[f"donor_{d}"] = (
            ids[meta.donor.values == tr],
            ids[meta.donor.values == val],
            ids[meta.donor.values == d],
        )
    trainval, test = train_test_split(
        ids, test_size=0.2, random_state=seed, stratify=meta.batch
    )
    train, val = train_test_split(
        trainval, test_size=0.2, random_state=seed, stratify=meta.iloc[trainval].batch
    )
    splits["random"] = (train, val, test)
    for parts in splits.values():
        assert sum(len(a) for a in parts) == len(meta)
        assert not (
            set(parts[0]) & set(parts[1])
            or set(parts[0]) & set(parts[2])
            or set(parts[1]) & set(parts[2])
        )
    return splits
