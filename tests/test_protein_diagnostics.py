import numpy as np
from citebench.metrics import protein_correlations


def test_constant_protein_is_undefined_not_negative_association():
    y = np.array([[1, 3], [2, 1], [4, 2]], dtype=float)
    p = np.column_stack([np.ones(3), y[:, 1]])
    r = protein_correlations(y, p)
    assert np.isnan(r[0])
    assert np.isclose(r[1], 1)
