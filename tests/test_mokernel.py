import lab as B
import pytest

from stheno.input import MultiInput
from stheno.kernel import EQ
from stheno.measure import Measure, GP
from stheno.mokernel import MultiOutputKernel
from .util import approx


def test_mokernel():
    m = Measure()
    p1 = GP(1 * EQ(), measure=m)
    p2 = GP(2 * EQ().stretch(2), measure=m)

    mok = MultiOutputKernel(m, p1, p2)
    ks = m.kernels

    x1 = B.linspace(0, 1, 10)
    x2 = B.linspace(1, 2, 5)
    x3 = B.linspace(1, 2, 10)

    # Check representation.
    assert str(mok) == "MultiOutputKernel(EQ(), 2 * (EQ() > 2))"

    # `B.Numeric` versus `B.Numeric`:
    approx(
        mok(x1, x2),
        B.concat2d(
            [ks[p1, p1](x1, x2), ks[p1, p2](x1, x2)],
            [ks[p2, p1](x1, x2), ks[p2, p2](x1, x2)],
        ),
    )
    approx(
        mok.elwise(x1, x3),
        B.concat(ks[p1, p1].elwise(x1, x3), ks[p2, p2].elwise(x1, x3), axis=0),
    )

    # `B.Numeric` versus `FDD`:
    approx(mok(p1(x1), x2), B.concat(ks[p1, p1](x1, x2), ks[p1, p2](x1, x2), axis=1))
    approx(mok(p2(x1), x2), B.concat(ks[p2, p1](x1, x2), ks[p2, p2](x1, x2), axis=1))
    approx(mok(x1, p1(x2)), B.concat(ks[p1, p1](x1, x2), ks[p2, p1](x1, x2), axis=0))
    approx(mok(x1, p2(x2)), B.concat(ks[p1, p2](x1, x2), ks[p2, p2](x1, x2), axis=0))
    with pytest.raises(ValueError):
        mok.elwise(x1, p2(x3))
    with pytest.raises(ValueError):
        mok.elwise(p1(x1), x3)

    # `FDD` versus `FDD`:
    approx(mok(p1(x1), p1(x2)), ks[p1](x1, x2))
    approx(mok(p1(x1), p2(x2)), ks[p1, p2](x1, x2))
    approx(mok.elwise(p1(x1), p1(x3)), ks[p1].elwise(x1, x3))
    approx(mok.elwise(p1(x1), p2(x3)), ks[p1, p2].elwise(x1, x3))

    # `MultiInput` versus `MultiInput`:
    approx(
        mok(MultiInput(p2(x1), p1(x2)), MultiInput(p2(x1))),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    with pytest.raises(ValueError):
        mok.elwise(MultiInput(p2(x1), p1(x3)), MultiInput(p2(x1)))
    approx(
        mok.elwise(MultiInput(p2(x1), p1(x3)), MultiInput(p2(x1), p1(x3))),
        B.concat(ks[p2, p2].elwise(x1, x1), ks[p1, p1].elwise(x3, x3), axis=0),
    )

    # `MultiInput` versus `FDD`:
    approx(
        mok(MultiInput(p2(x1), p1(x2)), p2(x1)),
        B.concat(ks[p2, p2](x1, x1), ks[p1, p2](x2, x1), axis=0),
    )
    approx(
        mok(p2(x1), MultiInput(p2(x1), p1(x2))),
        B.concat(ks[p2, p2](x1, x1), ks[p2, p1](x1, x2), axis=1),
    )
    with pytest.raises(ValueError):
        mok.elwise(MultiInput(p2(x1), p1(x3)), p2(x1))
    with pytest.raises(ValueError):
        mok.elwise(p2(x1), MultiInput(p2(x1), p1(x3)))
