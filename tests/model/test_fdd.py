import lab as B
import matrix
import numpy as np
from mlkernels import EQ, num_elements

from stheno.model import GP, FDD
from stheno.model.fdd import _noise_as_matrix


def test_noise_as_matrix():
    def check(noise, dtype, n, asserted_type):
        noise = _noise_as_matrix(noise, dtype, n)
        assert isinstance(noise, asserted_type)
        assert B.dtype(noise) == dtype
        assert B.shape(noise) == (n, n)

    check(None, int, 5, matrix.Zero)
    check(None, float, 5, matrix.Zero)
    check(1, np.int64, 5, matrix.Diagonal)
    check(1.0, np.float64, 5, matrix.Diagonal)
    check(B.ones(int, 5), np.int64, 5, matrix.Diagonal)
    check(B.ones(float, 5), np.float64, 5, matrix.Diagonal)
    check(matrix.Dense(B.ones(int, 5, 5)), np.int64, 5, matrix.Dense)
    check(matrix.Dense(B.randn(float, 5, 5)), np.float64, 5, matrix.Dense)


def test_fdd():
    p = GP(EQ())

    # Test specification without noise.
    for fdd in [p(1), FDD(p, 1)]:
        assert isinstance(fdd, FDD)
        assert fdd.x is 1
        assert fdd.p is p
        assert isinstance(fdd.noise, matrix.Zero)
        rep = (
            "<FDD:\n"
            " process=GP(0, EQ()),\n"
            " input=1,\n"
            " noise=<zero matrix: shape=1x1, dtype=int>>"
        )
        assert str(fdd) == rep
        assert repr(fdd) == rep

        # Check `dtype` and `num_elements`.
        assert B.dtype(fdd) == int
        assert num_elements(fdd) == 1

    # Test specification with noise.
    fdd = p(1.0, np.array([1, 2]))
    assert isinstance(fdd, FDD)
    assert fdd.x is 1.0
    assert fdd.p is p
    assert isinstance(fdd.noise, matrix.Diagonal)
    assert str(fdd) == (
        "<FDD:\n"
        " process=GP(0, EQ()),\n"
        " input=1.0,\n"
        " noise=<diagonal matrix: shape=2x2, dtype=int64>>"
    )
    assert repr(fdd) == (
        "<FDD:\n"
        " process=GP(0, EQ()),\n"
        " input=1.0,\n"
        " noise=<diagonal matrix: shape=2x2, dtype=int64\n"
        "        diag=[1 2]>>"
    )
    assert B.dtype(fdd) == float
    assert num_elements(fdd) == 1

    # Test construction with `id`.
    fdd = FDD(5, 1)
    assert fdd.p is 5
    assert fdd.x is 1
    assert fdd.noise is None
