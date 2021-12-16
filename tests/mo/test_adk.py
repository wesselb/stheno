import lab as B
from mlkernels import EQ, Linear, pairwise, elwise

from stheno import AmbiguousDimensionalityKernel as ADK, dimensionality
from ..util import approx


def test_adk():
    # Test properties.
    assert ADK(EQ()).stationary
    assert not ADK(Linear()).stationary

    # Test printing.
    assert str(ADK(EQ())) == "EQ()"
    assert str(ADK(EQ()) + ADK(Linear())) == "EQ() + Linear()"
    assert str(ADK(EQ() + Linear())) == "EQ() + Linear()"

    # Test equality.
    assert ADK(EQ()) == ADK(EQ())
    assert ADK(EQ()) != ADK(Linear())

    # Test computation.
    x = B.linspace(0, 5, 10)
    approx(pairwise(ADK(EQ()), x), pairwise(EQ(), x))
    approx(elwise(ADK(EQ()), x), elwise(EQ(), x))

    # Check that the dimensionality resolves to `None`.
    assert dimensionality(EQ()) == 1
    assert dimensionality(ADK(EQ())) is None
