import lab as B
import pytest
from mlkernels import num_elements, PosteriorKernel, SubspaceKernel

from stheno import (
    Measure,
    GP,
    EQ,
    MultiOutputKernel,
    infer_size,
    dimensionality,
    AmbiguousDimensionalityKernel as ADK,
)


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

    # Check that the dimensionality must be inferrable.
    assert infer_size(EQ(), x) == 5
    with pytest.raises(RuntimeError):
        infer_size(ADK(EQ()), x)


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

    # Check that dimensionalities must line up.
    with pytest.raises(RuntimeError):
        dimensionality(k1 + EQ())

    # Check `PosteriorKernel`.
    assert dimensionality(PosteriorKernel(EQ(), EQ(), EQ(), None, 0)) == 1
    assert dimensionality(PosteriorKernel(k1, k2, k2, None, 0)) == 2
    assert dimensionality(PosteriorKernel(k1, k2, ADK(EQ()), None, 0)) == 2
    with pytest.raises(RuntimeError):
        assert dimensionality(PosteriorKernel(k1, k2, EQ(), None, 0)) == 2

    # Check `SubspaceKernel`.
    assert dimensionality(SubspaceKernel(EQ(), EQ(), None, 0)) == 1
    assert dimensionality(SubspaceKernel(k1, k2, None, 0)) == 2
    assert dimensionality(SubspaceKernel(k1, ADK(EQ()), None, 0)) == 2
    with pytest.raises(RuntimeError):
        assert dimensionality(SubspaceKernel(k1, EQ(), None, 0)) == 2
