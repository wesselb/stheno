import lab as B
import matrix
import numpy as np
import pytest
from lab.shape import Dimension
from mlkernels import EQ, Exp, num_elements

from stheno import Measure, GP, FDD, cross
from stheno import infer_size
from stheno.model.fdd import _noise_as_matrix
from ..util import approx


def test_noise_as_matrix():
    def check(noise, dtype, n, asserted_type):
        noise = _noise_as_matrix(noise, dtype, n)
        assert isinstance(noise, asserted_type)
        assert B.dtype(noise) == dtype
        assert B.shape(noise) == (n, n)

    # Check that the type for `n` is appropriate.
    for d in [5, Dimension(5)]:
        check(None, int, d, matrix.Zero)
        check(None, float, d, matrix.Zero)
        check(1, np.int64, d, matrix.Diagonal)
        check(1.0, np.float64, d, matrix.Diagonal)
        check(B.ones(int, 5), np.int64, d, matrix.Diagonal)
        check(B.ones(float, 5), np.float64, d, matrix.Diagonal)
        check(matrix.Dense(B.ones(int, 5, 5)), np.int64, d, matrix.Dense)
        check(matrix.Dense(B.randn(float, 5, 5)), np.float64, d, matrix.Dense)


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
            " noise=<zero matrix: batch=(), shape=(1, 1), dtype=int>>"
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
        " noise=<diagonal matrix: batch=(), shape=(2, 2), dtype=int64>>"
    )
    assert repr(fdd) == (
        "<FDD:\n"
        " process=GP(0, EQ()),\n"
        " input=1.0,\n"
        " noise=<diagonal matrix: batch=(), shape=(2, 2), dtype=int64\n"
        "        diag=[1 2]>>"
    )
    assert B.dtype(fdd) == float
    assert num_elements(fdd) == 1

    # Test construction with `id`.
    fdd = FDD(5, 1)
    assert fdd.p is 5
    assert fdd.x is 1
    assert fdd.noise is None


def test_fdd_take():
    with Measure():
        f1 = GP(EQ())
        f2 = GP(Exp())
        f = cross(f1, f2)

    x = B.linspace(0, 3, 5)
    # Build an FDD with a very complicated input specification.
    fdd = f((x, (f2(x), x), f1(x), (f2(x), (f1(x), x))))
    n = infer_size(fdd.p.kernel, fdd.x)
    fdd = f(fdd.x, matrix.Diagonal(B.rand(n)))

    # Flip a coin for every element.
    mask = B.randn(n) > 0
    taken_fdd = B.take(fdd, mask)

    approx(taken_fdd.mean, B.take(fdd.mean, mask))
    approx(taken_fdd.var, B.submatrix(fdd.var, mask))
    approx(taken_fdd.noise, B.submatrix(fdd.noise, mask))
    assert isinstance(taken_fdd.noise, matrix.Diagonal)

    # Test that only masks are supported, for now.
    with pytest.raises(AssertionError):
        B.take(fdd, np.array([1, 2]))
