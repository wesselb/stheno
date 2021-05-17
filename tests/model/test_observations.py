import lab as B

from stheno import Measure, GP, EQ, Matern12, combine, Normal
from .util import assert_equal_normals
from ..util import approx


def test_combine():
    x1 = B.linspace(0, 2, 10)
    x2 = B.linspace(2, 4, 10)

    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Matern12(), measure=m)
    y1 = p1(x1).sample()
    y2 = p2(x2).sample()

    # Check the one-argument case.
    assert_equal_normals(combine(p1(x1, 1)), p1(x1, 1))
    fdd_combined, y_combined = combine((p1(x1, 1), B.squeeze(y1)))
    assert_equal_normals(fdd_combined, p1(x1, 1))
    approx(y_combined, y1)

    # Check the two-argument case.
    fdd_combined = combine(p1(x1, 1), p2(x2, 2))
    assert_equal_normals(
        fdd_combined,
        Normal(B.block_diag(p1(x1, 1).var, p2(x2, 2).var)),
    )
    fdd_combined, y_combined = combine((p1(x1, 1), B.squeeze(y1)), (p2(x2, 2), y2))
    assert_equal_normals(
        fdd_combined,
        Normal(B.block_diag(p1(x1, 1).var, p2(x2, 2).var)),
    )
    approx(y_combined, B.concat(y1, y2, axis=0))
