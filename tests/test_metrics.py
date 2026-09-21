import numpy as np
import pytest
from citebench.metrics import pearson_rows, paired_cluster_bootstrap


def test_exact_metric_and_affine_invariance():
    y = np.array([[1, 2, 4], [2, 0, 1.0]])
    assert np.allclose(pearson_rows(y, 3 * y + 5), 1)
    assert np.allclose(pearson_rows(y, -y), -1)
    assert np.all(pearson_rows(y, np.ones_like(y)) == -1)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        pearson_rows(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        pearson_rows(np.array([[0, 1, np.nan]]), np.ones((1, 3)))


def test_paired_bootstrap_identity():
    y = np.random.default_rng(0).normal(size=(12, 5))
    r = paired_cluster_bootstrap(y, y, y, np.repeat(["a", "b", "c"], 4), repeats=100)
    assert r["delta"] == r["low"] == r["high"] == 0
