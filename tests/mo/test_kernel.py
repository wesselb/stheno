import lab as B
import pytest
from mlkernels import EQ

from stheno.mo import MultiOutputKernel, dimensionality
from stheno.model import Measure, GP
from tests.util import approx


def test_mok():
    x1 = B.linspace(0, 1, 10)
    x2 = B.linspace(1, 2, 5)
    x3 = B.linspace(1, 2, 10)

    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(2 * EQ().stretch(2), measure=m)

    k = MultiOutputKernel(m, p1, p2)
    ks = m.kernels

    # Check dimensionality.
    assert dimensionality(k) == 2

    # Check representation.
    assert str(k) == "MultiOutputKernel(EQ(), 2 * (EQ() > 2))"

    # Input versus input:
    approx(
        k(x1, x2),
        B.concat2d(
            [ks[p1, p1](x1, x2), ks[p1, p2](x1, x2)],
            [ks[p2, p1](x1, x2), ks[p2, p2](x1, x2)],
        ),
    )
    approx(
        k.elwise(x1, x3),
        B.concat(ks[p1, p1].elwise(x1, x3), ks[p2, p2].elwise(x1, x3), axis=0),
    )

    # Input versus `FDD`:
    approx(k(p1(x1), x2), B.concat(ks[p1, p1](x1, x2), ks[p1, p2](x1, x2), axis=1))
    approx(k(p2(x1), x2), B.concat(ks[p2, p1](x1, x2), ks[p2, p2](x1, x2), axis=1))
    approx(k(x1, p1(x2)), B.concat(ks[p1, p1](x1, x2), ks[p2, p1](x1, x2), axis=0))
    approx(k(x1, p2(x2)), B.concat(ks[p1, p2](x1, x2), ks[p2, p2](x1, x2), axis=0))
    with pytest.raises(ValueError):
        k.elwise(x1, p2(x3))
    with pytest.raises(ValueError):
        k.elwise(p1(x1), x3)

    # `FDD` versus `FDD`:
    approx(k(p1(x1), p1(x2)), ks[p1](x1, x2))
    approx(k(p1(x1), p2(x2)), ks[p1, p2](x1, x2))
    approx(k.elwise(p1(x1), p1(x3)), ks[p1].elwise(x1, x3))
    approx(k.elwise(p1(x1), p2(x3)), ks[p1, p2].elwise(x1, x3))

    # Multiple inputs versus input:
    approx(
        k((p2(x1), p1(x2)), x1),
        B.concat2d(
            [ks[p2, p1](x1, x1), ks[p2, p2](x1, x1)],
            [ks[p1, p1](x2, x1), ks[p1, p2](x2, x1)],
        ),
    )
    approx(
        k(x1, (p2(x1), p1(x2))),
        B.concat2d(
            [ks[p1, p2](x1, x1), ks[p1, p1](x1, x2)],
            [ks[p2, p2](x1, x1), ks[p2, p1](x1, x2)],
        ),
    )
    with pytest.raises(ValueError):
        k.elwise((p2(x1), p1(x3)), p2(x1))
    with pytest.raises(ValueError):
        k.elwise(p2(x1), (p2(x1), p1(x3)))

    # Multiple inputs versus `FDD`:
    approx(
        k((p2(x1), p1(x2)), p2(x1)),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    approx(
        k(p2(x1), (p2(x1), p1(x2))),
        B.concat(ks[p2, p2](x1, x1), ks[p2, p1](x1, x2), axis=1),
    )
    with pytest.raises(ValueError):
        k.elwise((p2(x1), p1(x3)), p2(x1))
    with pytest.raises(ValueError):
        k.elwise(p2(x1), (p2(x1), p1(x3)))

    # Multiple inputs versus multiple inputs:
    approx(
        k((p2(x1), p1(x2)), (p2(x1))),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    with pytest.raises(ValueError):
        k.elwise((p2(x1), p1(x3)), (p2(x1),))
    approx(
        k.elwise((p2(x1), p1(x3)), (p2(x1), p1(x3))),
        B.concat(ks[p2, p2].elwise(x1, x1), ks[p1, p1].elwise(x3, x3), axis=0),
    )
