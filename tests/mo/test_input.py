import lab as B
import pytest
from mlkernels import num_elements

from stheno import Measure, GP, EQ, MultiOutputKernel, infer_size, dimensionality


def test_num_elements():
    assert num_elements((1, B.ones(5))) == 6


def test_infer_size():
    x = B.linspace(0, 2, 5)

    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(2 * EQ().stretch(2), measure=m)
    k = MultiOutputKernel(m, p1, p2)

    assert infer_size(k, x) == 10
    assert infer_size(k, p1(x)) == 5
    assert infer_size(k, (x, p1(x))) == 15


def test_dimensionality():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(2 * EQ().stretch(2), measure=m)

    k1 = MultiOutputKernel(m, p1, p2)
    k2 = MultiOutputKernel(m, p1, p1)

    assert dimensionality(EQ()) == 1
    assert dimensionality(k1) == 2

    # Test the unpacking of `Wrapped`s and `Join`s.
    assert dimensionality(k1 + k2) == 2
    assert dimensionality(k1 * k2) == 2
    assert dimensionality(k1.periodic(1)) == 2
    assert dimensionality(k1.stretch(2)) == 2

    # Test consistency check.
    with pytest.raises(RuntimeError):
        dimensionality(k1 + EQ())
