import lab as B

from stheno.measure import Measure, GP
from stheno.input import MultiInput
from stheno.kernel import EQ
from stheno.momean import MultiOutputMean
from .util import approx


def test_momean():
    m = Measure()
    p1 = GP(lambda x: 2 * x, 1 * EQ(), measure=m)
    p2 = GP(1, 2 * EQ().stretch(2), measure=m)

    mom = MultiOutputMean(m, p1, p2)
    ms = m.means

    # Check representation.
    assert str(mom) == "MultiOutputMean(<lambda>, 1)"

    # Check computation.
    x = B.linspace(0, 1, 10)
    approx(mom(x), B.concat(ms[p1](x), ms[p2](x), axis=0))
    approx(mom(p1(x)), ms[p1](x))
    approx(mom(p2(x)), ms[p2](x))
    approx(mom(MultiInput(p2(x), p1(x))), B.concat(ms[p2](x), ms[p1](x), axis=0))
