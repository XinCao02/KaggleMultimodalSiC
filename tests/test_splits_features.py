import numpy as np
import pandas as pd
from citebench.data import make_splits
from citebench.features import RNAFeatures


def test_no_overlap_and_temporal_order():
    meta = pd.DataFrame(
        [
            {"donor": d, "day": day, "batch": f"{d}_{day}"}
            for d in [13176, 31800, 32606]
            for day in [2, 3, 4]
            for _ in range(20)
        ]
    )
    splits = make_splits(meta)
    tr, va, te = splits["temporal"]
    assert meta.iloc[tr].day.max() < meta.iloc[va].day.min() < meta.iloc[te].day.min()
    for name, (tr, va, te) in splits.items():
        assert len(set(tr) | set(va) | set(te)) == len(meta)
        if name.startswith("donor"):
            assert not set(meta.iloc[tr].donor) & set(meta.iloc[te].donor)


def test_heldout_targets_cannot_change_features():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, (60, 20)).astype("float32")
    y = rng.normal(size=(60, 5)).astype("float32")
    meta = pd.DataFrame({"batch": np.repeat(["a", "b", "c"], 20)})
    tr = np.arange(40)
    genes = np.array([f"G{i}" for i in range(20)])
    a = RNAFeatures(n_hvg=15, n_components=4, top_per_target=1, device="cpu").fit(
        x, y, tr, meta, genes
    )
    modified = y.copy()
    modified[40:] *= 1000
    b = RNAFeatures(n_hvg=15, n_components=4, top_per_target=1, device="cpu").fit(
        x, modified, tr, meta, genes
    )
    assert np.array_equal(a.selected, b.selected)
    assert np.allclose(a.transform(x, tr), b.transform(x, tr))
