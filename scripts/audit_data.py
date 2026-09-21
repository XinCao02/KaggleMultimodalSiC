"""Verify content duplicates using hashes only as candidates, then exact equality."""

import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd


def main():
    d = Path("data/processed")
    out = Path("results")
    out.mkdir(exist_ok=True)
    x = np.load(d / "x_train.npy", mmap_mode="r")
    z = np.load(d / "x_test.npy", mmap_mode="r")
    a = pd.read_csv(d / "meta_train.csv")
    b = pd.read_csv(d / "meta_test.csv")
    seen = {}
    dup = 0
    for i, row in enumerate(x):
        key = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        previous = seen.setdefault(key, [])
        dup += any(np.array_equal(row, x[j]) for j in previous)
        previous.append(i)
    pairs = []
    for i, row in enumerate(z):
        key = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        for j in seen.get(key, []):
            if np.array_equal(row, x[j]):
                pairs.append(
                    {
                        "train_cell_id": a.cell_id.iloc[j],
                        "test_cell_id": b.cell_id.iloc[i],
                        "train_donor": int(a.donor.iloc[j]),
                        "test_donor": int(b.donor.iloc[i]),
                        "train_day": int(a.day.iloc[j]),
                        "test_day": int(b.day.iloc[i]),
                    }
                )
    matched = len(set(p["test_cell_id"] for p in pairs))
    result = {
        "train_cells": len(x),
        "test_cells": len(z),
        "train_duplicate_profiles": int(dup),
        "identical_train_test_profiles": matched,
        "official_scored_cite_cells": len(z) - matched,
        "method": "BLAKE2b candidates verified by exact array equality; float32 original normalized RNA profiles",
        "id_intersection": len(set(a.cell_id) & set(b.cell_id)),
    }
    (out / "data_audit.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(pairs).to_csv(out / "duplicate_cells.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
