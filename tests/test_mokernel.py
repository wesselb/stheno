import lab as B
import pytest

from stheno.input import Input, MultiInput
from stheno.kernel import EQ
from stheno.measure import Measure, GP
from stheno.mokernel import MultiOutputKernel
from .util import approx


@pytest.mark.parametrize("x1", [B.linspace(0, 1, 10), Input(B.linspace(0, 1, 10))])
@pytest.mark.parametrize("x2", [B.linspace(1, 2, 5), Input(B.linspace(0, 2, 5))])
@pytest.mark.parametrize("x3", [B.linspace(1, 2, 10), Input(B.linspace(1, 2, 10))])
def test_mokernel(x1, x2, x3):
    m = Measure()
    p1 = GP(1 * EQ(), measure=m)
    p2 = GP(2 * EQ().stretch(2), measure=m)

    k = MultiOutputKernel(m, p1, p2)
    ks = m.kernels

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

    # `MultiInput` versus input:
    approx(
        k(MultiInput(p2(x1), p1(x2)), x1),
        B.concat2d(
            [ks[p2, p1](x1, x1), ks[p2, p2](x1, x1)],
            [ks[p1, p1](x2, x1), ks[p1, p2](x2, x1)],
        ),
    )
    approx(
        k(x1, MultiInput(p2(x1), p1(x2))),
        B.concat2d(
            [ks[p1, p2](x1, x1), ks[p1, p1](x1, x2)],
            [ks[p2, p2](x1, x1), ks[p2, p1](x1, x2)],
        ),
    )
    with pytest.raises(ValueError):
        k.elwise(MultiInput(p2(x1), p1(x3)), p2(x1))
    with pytest.raises(ValueError):
        k.elwise(p2(x1), MultiInput(p2(x1), p1(x3)))

    # `MultiInput` versus `FDD`:
    approx(
        k(MultiInput(p2(x1), p1(x2)), p2(x1)),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    approx(
        k(p2(x1), MultiInput(p2(x1), p1(x2))),
        B.concat(ks[p2, p2](x1, x1), ks[p2, p1](x1, x2), axis=1),
    )
    with pytest.raises(ValueError):
        k.elwise(MultiInput(p2(x1), p1(x3)), p2(x1))
    with pytest.raises(ValueError):
        k.elwise(p2(x1), MultiInput(p2(x1), p1(x3)))

    # `MultiInput` versus `MultiInput`:
    approx(
        k(MultiInput(p2(x1), p1(x2)), MultiInput(p2(x1))),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    with pytest.raises(ValueError):
        k.elwise(MultiInput(p2(x1), p1(x3)), MultiInput(p2(x1)))
    approx(
        k.elwise(MultiInput(p2(x1), p1(x3)), MultiInput(p2(x1), p1(x3))),
        B.concat(ks[p2, p2].elwise(x1, x1), ks[p1, p1].elwise(x3, x3), axis=0),
    )
