
from tests.util import approx


def assert_equal_gps(p1, p2):
    assert p1.mean == p2.mean
    assert p1.kernel == p2.kernel


def assert_equal_normals(d1, d2):
    approx(d1.mean, d2.mean)
    approx(d1.var, d2.var)


def assert_equal_measures(fdds, post_ref, *posts):
    for post in posts:
        for fdd in fdds:
            assert_equal_normals(post_ref(fdd), post(fdd))

