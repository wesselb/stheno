import pytest
import lab as B

from stheno.measure import Measure, GP
from stheno.input import Input, MultiInput
from stheno.kernel import EQ
from stheno.momean import MultiOutputMean
from .util import approx


@pytest.mark.parametrize("x", [B.linspace(0, 1, 10), Input(B.linspace(0, 1, 10))])
def test_momean(x):
    prior = Measure()
    p1 = GP(lambda x: 2 * x, 1 * EQ(), measure=prior)
    p2 = GP(1, 2 * EQ().stretch(2), measure=prior)

    m = MultiOutputMean(prior, p1, p2)
    ms = prior.means

    # Check representation.
    assert str(m) == "MultiOutputMean(<lambda>, 1)"

    # Check computation.
    approx(m(x), B.concat(ms[p1](x), ms[p2](x), axis=0))
    approx(m(p1(x)), ms[p1](x))
    approx(m(p2(x)), ms[p2](x))
    approx(m(MultiInput(p2(x), p1(x))), B.concat(ms[p2](x), ms[p1](x), axis=0))
