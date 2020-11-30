import numpy as np
import pytest
import tensorflow as tf
from lab import B
from plum import NotFoundLookupError

from stheno.input import Input
from stheno.kernel import (
    Linear,
    EQ,
    Delta,
    Exp,
    ZeroKernel,
    OneKernel,
    ScaledKernel,
)
from stheno.mean import TensorProductMean, ZeroMean, ScaledMean, OneMean
from stheno.measure import Measure, GP, Obs, SparseObs, FDD, cross
from stheno.random import Normal
from .util import approx


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


def test_corner_cases():
    p1 = GP(EQ())
    p2 = GP(EQ())
    x = B.randn(10, 2)

    # Test check for measure group.
    with pytest.raises(AssertionError):
        p1 + p2
    with pytest.raises(AssertionError):
        p1 * p2

    # Test incompatible operations.
    with pytest.raises(NotFoundLookupError):
        p1 + p1(x)
    with pytest.raises(NotFoundLookupError):
        p1 * p1(x)

    # Check display of GPs.
    assert str(GP()) == "GP()"
    assert str(GP(EQ())) == "GP(0, EQ())"

    # Check test for prior.
    with pytest.raises(RuntimeError):
        GP().measure


def test_construction():
    p = GP(EQ())

    x = B.randn(10, 1)

    p.mean(x)
    p.mean(Input(x))

    p.kernel(x)
    p.kernel(Input(x))
    p.kernel(x, x)
    p.kernel(Input(x), x)
    p.kernel(x, Input(x))
    p.kernel(Input(x), Input(x))

    p.kernel.elwise(x)
    p.kernel.elwise(Input(x))
    p.kernel.elwise(x, x)
    p.kernel.elwise(Input(x), x)
    p.kernel.elwise(x, Input(x))
    p.kernel.elwise(Input(x), Input(x))

    # Test resolution of kernel and mean.
    k = EQ()
    m = TensorProductMean(lambda x: x ** 2)

    assert isinstance(GP(k).mean, ZeroMean)
    assert isinstance(GP(5, k).mean, ScaledMean)
    assert isinstance(GP(1, k).mean, OneMean)
    assert isinstance(GP(0, k).mean, ZeroMean)
    assert isinstance(GP(m, k).mean, TensorProductMean)
    assert isinstance(GP(k).kernel, EQ)
    assert isinstance(GP(5).kernel, ScaledKernel)
    assert isinstance(GP(1).kernel, OneKernel)
    assert isinstance(GP(0).kernel, ZeroKernel)

    # Test construction of finite-dimensional distribution.
    d = GP(m, k)(x)
    approx(d.var, k(x))
    approx(d.mean, m(x))


def test_sum_other():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())

    def five(y):
        return 5 * B.ones(B.shape(y)[0], 1)

    x = B.randn(5, 1)
    for p_sum in [
        # Add a numeric thing.
        p + 5.0,
        5.0 + p,
        p.measure.sum(GP(), p, 5.0),
        p.measure.sum(GP(), 5.0, p),
        # Add a function.
        p + five,
        five + p,
        p.measure.sum(GP(), p, five),
        p.measure.sum(GP(), five, p),
    ]:
        approx(p.mean(x) + 5.0, p_sum.mean(x))
        approx(p.mean(x) + 5.0, p_sum.mean(x))
        approx(p.kernel(x), p_sum.kernel(x))
        approx(p.kernel(x), p_sum.kernel(x))

    # Check that a `GP` cannot be summed with a `Normal`.
    with pytest.raises(NotFoundLookupError):
        p + Normal(np.eye(3))
    with pytest.raises(NotFoundLookupError):
        Normal(np.eye(3)) + p


def test_mul_other():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())

    def five(y):
        return 5 * B.ones(B.shape(y)[0], 1)

    x = B.randn(5, 1)
    for p_mul in [
        # Multiply numeric thing.
        p * 5.0,
        5.0 * p,
        p.measure.mul(GP(), p, 5.0),
        p.measure.mul(GP(), 5.0, p),
        # Multiply with a function.
        p * five,
        five * p,
        p.measure.mul(GP(), p, five),
        p.measure.mul(GP(), five, p),
    ]:
        approx(5.0 * p.mean(x), p_mul.mean(x))
        approx(5.0 * p.mean(x), p_mul.mean(x))
        approx(25.0 * p.kernel(x), p_mul.kernel(x))
        approx(25.0 * p.kernel(x), p_mul.kernel(x))

    # Check that a `GP` cannot be multiplied with a `Normal`.
    with pytest.raises(NotFoundLookupError):
        p * Normal(np.eye(3))
    with pytest.raises(NotFoundLookupError):
        Normal(np.eye(3)) * p


def test_fdd():
    p = GP(EQ())

    # Test construction.
    fdd = p(1)
    assert isinstance(fdd, FDD)
    assert fdd.x is 1
    assert fdd.p is p

    class A:
        def __str__(self):
            return "str"

        def __repr__(self):
            return "repr"

    # Check representation.
    assert str(p(A())) == "FDD(GP(0, EQ()), str)"
    assert repr(p(A())) == "FDD(GP(0, EQ()), repr)"


def test_measure_groups():
    prior = Measure()
    f1 = GP(EQ(), measure=prior)
    f2 = GP(EQ(), measure=prior)

    assert f1._measures == f2._measures

    x = B.linspace(0, 5, 10)
    y = f1(x).sample()

    post = prior | (f1(x), y)

    assert f1._measures == f2._measures == [prior, post]

    # Further extend the prior.

    f_sum = f1 + f2
    assert f_sum._measures == [prior, post]

    f3 = GP(EQ(), measure=prior)
    f_sum = f1 + f3
    assert f3._measures == f_sum._measures == [prior]

    with pytest.raises(AssertionError):
        post(f1) + f3

    # Extend the posterior.

    f_sum = post(f1) + post(f2)
    assert f_sum._measures == [post]

    f3 = GP(EQ(), measure=post)
    f_sum = post(f1) + f3
    assert f3._measures == f_sum._measures == [post]

    with pytest.raises(AssertionError):
        f1 + f3


def test_shorthands():
    p = GP(EQ())
    # Test shorthands for stretching and selection.
    assert_equal_gps(p > 2, p.stretch(2))
    assert_equal_gps(p[0], p.select(0))


def test_stationarity():
    m = Measure()

    p1 = GP(EQ(), measure=m)
    p2 = GP(EQ().stretch(2), measure=m)
    p3 = GP(EQ().periodic(10), measure=m)

    p = p1 + 2 * p2

    assert p.stationary

    p = p3 + p

    assert p.stationary

    p = p + GP(Linear(), measure=m)

    assert not p.stationary


def test_marginals():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    x = B.linspace(0, 5, 10)

    # Check that `marginals` outputs the right thing.
    mean, lower, upper = p(x).marginals()
    var = B.diag(p.kernel(x))
    approx(mean, p.mean(x)[:, 0])
    approx(lower, p.mean(x)[:, 0] - 1.96 * var ** 0.5)
    approx(upper, p.mean(x)[:, 0] + 1.96 * var ** 0.5)

    # Test correctness.
    y = p(x).sample()
    post = p.measure | (p(x), y)

    mean, lower, upper = post(p)(x).marginals()
    approx(mean, y[:, 0])
    approx(upper - lower, B.zeros(10), atol=1e-5)

    mean, lower, upper = post(p)(x + 100).marginals()
    approx(mean, p.mean(x + 100)[:, 0])
    approx(upper - lower, 3.92 * B.ones(10))


def test_conditioning():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
    p_sum = p1 + p2

    # Sample some data to condition on.
    x1 = B.linspace(0, 2, 2)
    y1 = p1(x1).sample()
    x_sum = B.linspace(3, 5, 3)
    y_sum = p_sum(x_sum).sample()

    # Determine FDDs to check.
    x_check = B.linspace(0, 5, 5)
    fdds_check = [
        cross(p1, p2, p_sum)(x_check),
        p1(x_check),
        p2(x_check),
        p_sum(x_check),
    ]

    assert_equal_measures(
        fdds_check,
        m.condition(p_sum(x_sum), y_sum),
        m.condition((p_sum(x_sum), y_sum)),
        m | (p_sum(x_sum), y_sum),
        m | ((p_sum(x_sum), y_sum),),
        m | Obs(p_sum(x_sum), y_sum),
        m | Obs((p_sum(x_sum), y_sum)),
    )

    assert_equal_measures(
        fdds_check,
        m.condition((p1(x1), y1), (p_sum(x_sum), y_sum)),
        m | ((p1(x1), y1), (p_sum(x_sum), y_sum)),
        m | Obs((p1(x1), y1), (p_sum(x_sum), y_sum)),
    )


def test_conditioning_prior():
    p = GP(EQ())
    x = B.zeros(0, 1)
    y = B.zeros(0, 1)
    post = p.measure | (p(x), y)
    assert post(p).mean is p.mean
    assert post(p).kernel is p.kernel


def test_sparse_conditioning_and_elbo():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
    e = GP(Delta(), measure=m)
    p_sum = p1 + p2

    # Sample some data to condition on.
    x1 = B.linspace(0, 2, 2)
    y1 = (p1 + e)(x1).sample()
    x_sum = B.linspace(3, 5, 3)
    y_sum = (p_sum + e)(x_sum).sample()

    # Determine FDDs to check.
    x_check = B.linspace(0, 5, 5)
    fdds_check = [
        cross(p1, p2, p_sum)(x_check),
        p1(x_check),
        p2(x_check),
        p_sum(x_check),
    ]

    # Check conditioning and ELBO on one data set.
    assert_equal_measures(
        fdds_check,
        m | ((p_sum + e)(x_sum), y_sum),
        m | SparseObs(p_sum(x_sum), e, p_sum(x_sum), y_sum),
        m | SparseObs((p_sum(x_sum),), e, p_sum(x_sum), y_sum),
        m | SparseObs((p_sum(x_sum), p1(x1)), e, p_sum(x_sum), y_sum),
        m | SparseObs(p_sum(x_sum), (e, p_sum(x_sum), y_sum)),
        m | SparseObs((p_sum(x_sum),), (e, p_sum(x_sum), y_sum)),
        m.condition(
            SparseObs(
                (p_sum(x_sum), p1(x1)),
                (e, p_sum(x_sum), y_sum),
            )
        ),
    )
    approx(
        m.logpdf(Obs((p_sum + e)(x_sum), y_sum)),
        SparseObs(p_sum(x_sum), (e, p_sum(x_sum), y_sum)).elbo(m),
    )

    # Check conditioning and ELBO on two data sets.
    assert_equal_measures(
        fdds_check,
        m | (((p_sum + e)(x_sum), y_sum), ((p1 + e)(x1), y1)),
        m.condition(
            SparseObs((p_sum(x_sum), p1(x1)), (e, p_sum(x_sum), y_sum), (e, p1(x1), y1))
        ),
    )
    approx(
        m.logpdf(Obs(((p_sum + e)(x_sum), y_sum), ((p1 + e)(x1), y1))),
        SparseObs(
            (p_sum(x_sum), p1(x1)), (e, p_sum(x_sum), y_sum), (e, p1(x1), y1)
        ).elbo(m),
    )

    # The following lose information, so check them separately.
    assert_equal_measures(
        fdds_check,
        m | SparseObs(p_sum(x_sum), (e, p_sum(x_sum), y_sum), (e, p1(x1), y1)),
        m | SparseObs((p_sum(x_sum),), (e, p_sum(x_sum), y_sum), (e, p1(x1), y1)),
    )

    # Test lazy computation.
    obs = SparseObs(p_sum(x_sum), e, p_sum(x_sum), y_sum)
    for name in ["K_z", "elbo", "mu", "A"]:
        approx(
            getattr(SparseObs(p_sum(x_sum), e, p_sum(x_sum), y_sum), name)(m),
            getattr(obs, name)(m),
        )

    # Test requirement that noise must be diagonal.
    with pytest.raises(RuntimeError):
        SparseObs(p_sum(x_sum), p_sum, p_sum(x_sum), y_sum).elbo(m)


def test_logpdf():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
    e = GP(Delta(), measure=m)
    p3 = p1 + p2

    x1 = B.linspace(0, 2, 5)
    x2 = B.linspace(1, 3, 6)
    x3 = B.linspace(2, 4, 7)
    y1, y2, y3 = m.sample(p1(x1), p2(x2), p3(x3))

    # Test case that only one process is fed.
    approx(p1(x1).logpdf(y1), m.logpdf(p1(x1), y1))
    approx(p1(x1).logpdf(y1), m.logpdf((p1(x1), y1)))

    # Compute the logpdf with the product rule.
    d1 = m
    d2 = d1 | (p1(x1), y1)
    d3 = d2 | (p2(x2), y2)
    approx(
        d1(p1)(x1).logpdf(y1) + d2(p2)(x2).logpdf(y2) + d3(p3)(x3).logpdf(y3),
        m.logpdf((p1(x1), y1), (p2(x2), y2), (p3(x3), y3)),
    )

    # Check that `Measure.logpdf` allows `Obs` and `SparseObs`.
    obs = Obs(p3(x3), y3)
    approx(m.logpdf(obs), p3(x3).logpdf(y3))
    obs = SparseObs(p3(x3), e, p3(x3), y3)
    approx(m.logpdf(obs), (p3 + e)(x3).logpdf(y3))


def test_stretching():
    # Test construction:
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    assert str(p.stretch(1)) == "GP(<lambda> > 1, EQ() > 1)"

    # Test case:
    p = GP(EQ())
    p_stretched = p.stretch(5)

    x = B.linspace(0, 5, 10)
    y = p_stretched(x).sample()

    post = p.measure | (p_stretched(x), y)
    assert_equal_normals(post(p(x / 5)), post(p_stretched(x)))
    assert_equal_normals(post(p(x)), post(p_stretched(x * 5)))


def test_shifting():
    # Test construction:
    p = GP(TensorProductMean(lambda x: x ** 2), Linear())
    assert str(p.shift(1)) == "GP(<lambda> shift 1, Linear() shift 1)"

    # Test case:
    p = GP(EQ())
    p_shifted = p.shift(5)

    x = B.linspace(0, 5, 10)
    y = p_shifted(x).sample()

    post = p.measure | (p_shifted(x), y)
    assert_equal_normals(post(p(x - 5)), post(p_shifted(x)))
    assert_equal_normals(post(p(x)), post(p_shifted(x + 5)))


def test_input_transform():
    # Test construction:
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    assert (
        str(p.transform(lambda x: x))
        == "GP(<lambda> transform <lambda>, EQ() transform <lambda>)"
    )

    # Test case:
    p = GP(EQ())
    p_transformed = p.transform(lambda x: B.sqrt(x))

    x = B.linspace(0, 5, 10)
    y = p_transformed(x).sample()

    post = p.measure | (p_transformed(x), y)
    assert_equal_normals(post(p(B.sqrt(x))), post(p_transformed(x)))
    assert_equal_normals(post(p(x)), post(p_transformed(x * x)))


def test_selection():
    # Test construction:
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    assert str(p.select(1)) == "GP(<lambda> : [1], EQ() : [1])"
    assert str(p.select(1, 2)) == "GP(<lambda> : [1, 2], EQ() : [1, 2])"

    # Test case:
    p = GP(EQ())  # 1D
    p2 = p.select(0)  # 2D

    x = B.linspace(0, 5, 10)
    x21 = B.stack(x, B.randn(10), axis=1)
    x22 = B.stack(x, B.randn(10), axis=1)
    y = p2(x).sample()

    post = p.measure | (p2(x21), y)
    approx(post(p(x)).mean, y)
    assert_equal_normals(post(p(x)), post(p2(x21)))

    post = p.measure | (p2(x22), y)
    approx(post(p(x)).mean, y)
    assert_equal_normals(post(p(x)), post(p2(x22)))

    post = p.measure | (p(x), y)
    approx(post(p2(x21)).mean, y)
    approx(post(p2(x22)).mean, y)
    assert_equal_normals(post(p2(x21)), post(p(x)))
    assert_equal_normals(post(p2(x22)), post(p(x)))


def test_derivative():
    # Test construction:
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    assert str(p.diff(1)) == "GP(d(1) <lambda>, d(1) EQ())"

    # Test case:
    p = GP(EQ())
    dp = p.diff()

    x = B.linspace(tf.float64, 0, 1, 100)
    y = 2 * x

    x_check = B.linspace(tf.float64, 0.2, 0.8, 100)

    # Test conditioning on function.
    post = p.measure | (p(x), y)
    approx(post(dp)(x_check).mean, 2 * B.ones(100, 1), atol=1e-4)

    # Test conditioning on derivative.
    zero = B.cast(tf.float64, 0)
    post = p.measure | ((p(zero), zero), (dp(x), y))
    approx(post(p)(x_check).mean, x_check[:, None] ** 2, atol=1e-4)


def test_multi_sample():
    m = Measure()
    p1 = GP(1, 0, measure=m)
    p2 = GP(2, 0, measure=m)
    p3 = GP(3, 0, measure=m)

    x1 = B.linspace(0, 1, 5)
    x2 = B.linspace(0, 1, 10)
    x3 = B.linspace(0, 1, 15)

    s1, s2, s3 = m.sample(p1(x1), p2(x2), p3(x3))

    assert B.shape(p1(x1).sample()) == s1.shape == (5, 1)
    assert B.shape(p2(x2).sample()) == s2.shape == (10, 1)
    assert B.shape(p3(x3).sample()) == s3.shape == (15, 1)

    approx(s1, 1 * B.ones(5, 1))
    approx(s2, 2 * B.ones(10, 1))
    approx(s3, 3 * B.ones(15, 1))


def test_approximate_multiplication():
    m = Measure()
    p1 = GP(20, EQ(), measure=m)
    p2 = GP(20, EQ(), measure=m)
    p_prod = p1 * p2

    # Sample functions.
    x = B.linspace(0, 10, 50)
    s1, s2 = m.sample(p1(x), p2(x))

    # Perform product.
    post = m | ((p1(x), s1), (p2(x), s2))
    approx(post(p_prod)(x).mean, s1 * s2, rtol=1e-2)

    # Perform division.
    cur_epsilon = B.epsilon
    B.epsilon = 1e-8
    post = m | ((p1(x), s1), (p_prod(x), s1 * s2))
    approx(post(p2)(x).mean, s2, rtol=1e-2)
    B.epsilon = cur_epsilon


def test_naming():
    m = Measure()

    p1 = GP(EQ(), 1, measure=m)
    p2 = GP(EQ(), 2, measure=m)

    # Test setting and getting names.
    p1.name = "name"

    assert m["name"] is p1
    assert p1.name == "name"
    assert m[p1] == "name"
    with pytest.raises(KeyError):
        m["other_name"]
    with pytest.raises(KeyError):
        m[p2]

    # Check that names can not be doubly assigned.
    def doubly_assign():
        p2.name = "name"

    with pytest.raises(RuntimeError):
        doubly_assign()

    # Move name to other GP.
    p1.name = "other_name"
    p2.name = "name"

    # Check that everything has been properly assigned.
    assert m["name"] is p2
    assert p2.name == "name"
    assert m[p2] == "name"
    assert m["other_name"] is p1
    assert p1.name == "other_name"
    assert m[p1] == "other_name"

    # Test giving a name to the constructor.
    p3 = GP(EQ(), name="yet_another_name", measure=m)
    assert m["yet_another_name"] is p3
    assert p3.name == "yet_another_name"
    assert m[p3] == "yet_another_name"


def test_formatting():
    p = 2 * GP(1, EQ(), measure=Measure())
    assert str(p.display(lambda x: x ** 2)) == "GP(4 * 1, 16 * EQ())"


def test_case_summation_with_itself():
    p = GP(EQ())
    p_many = p + p + p + p + p

    x = B.linspace(0, 10, 5)
    approx(p_many(x).var, 25 * p(x).var)
    approx(p_many(x).mean, B.zeros(5, 1))

    y = B.randn(5, 1)
    post = p.measure | (p(x), y)
    approx(post(p_many)(x).mean, 5 * y)


def test_case_additive_model():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(EQ(), measure=m)
    p_sum = p1 + p2

    x = B.linspace(0, 5, 10)
    y1 = p1(x).sample()
    y2 = p2(x).sample()

    # First, test independence:
    assert m.kernels[p2, p1] == ZeroKernel()
    assert m.kernels[p1, p2] == ZeroKernel()

    # Now run through some test cases:
    post = (m | (p1(x), y1)) | (p2(x), y2)
    approx(post(p_sum)(x).mean, y1 + y2)

    post = (m | (p2(x), y2)) | (p1(x), y1)
    approx(post(p_sum)(x).mean, y1 + y2)

    post = (m | (p1(x), y1)) | (p_sum(x), y1 + y2)
    approx(post(p2)(x).mean, y2)

    post = (m | (p_sum(x), y1 + y2)) | (p1(x), y1)
    approx(post(p2)(x).mean, y2)

    post = (m | (p2(x), y2)) | (p_sum(x), y1 + y2)
    approx(post(p1)(x).mean, y1)

    post = (m | (p_sum(x), y1 + y2)) | (p2(x), y2)
    approx(post(p1)(x).mean, y1)


def test_case_fd_derivative():
    x = B.linspace(0, 10, 50)
    y = np.sin(x)

    p = GP(0.7 * EQ().stretch(1.0))
    dp = (p.shift(-1e-3) - p.shift(1e-3)) / 2e-3

    post = p.measure | (p(x), y)
    approx(post(dp)(x).mean, np.cos(x)[:, None], atol=1e-4)


def test_case_reflection():
    p = GP(EQ())
    p2 = 5 - p

    x = B.linspace(0, 5, 10)
    y = p(x).sample()

    post = p.measure | (p(x), y)
    approx(post(p2)(x).mean, 5 - y)

    post = p.measure | (p2(x), 5 - y)
    approx(post(p)(x).mean, y)


def test_case_negation():
    p = GP(EQ())
    p2 = -p

    x = B.linspace(0, 5, 10)
    y = p(x).sample()

    post = p.measure | (p(x), y)
    approx(post(p2)(x).mean, -y)

    post = p.measure | (p2(x), -y)
    approx(post(p)(x).mean, y)


def test_case_approximate_derivative():
    p = GP(EQ().stretch(1.0))
    dp = p.diff_approx()

    x = B.linspace(0, 1, 100)
    y = 2 * x

    x_check = B.linspace(0.2, 0.8, 100)

    # Test conditioning on function.
    post = p.measure | (p(x), y)
    approx(post(dp)(x_check).mean, 2 * B.ones(100, 1), atol=1e-3)

    # Test conditioning on derivative.
    orig_epsilon = B.epsilon
    B.epsilon = 1e-10
    post = p.measure | ((p(0), 0), (dp(x), y))
    approx(post(p)(x_check).mean, x_check[:, None] ** 2, atol=1e-3)
    B.epsilon = orig_epsilon


def test_case_blr():
    m = Measure()
    x = B.linspace(0, 10, 100)

    slope = GP(1, measure=m)
    intercept = GP(1, measure=m)
    f = slope * (lambda x: x) + intercept
    y = f + 1e-2 * GP(Delta(), measure=m)

    # Sample observations, true slope, and intercept.
    y_obs, true_slope, true_intercept = m.sample(y(x), slope(0), intercept(0))

    # Predict.
    post = m | (y(x), y_obs)
    approx(post(slope)(0).mean, true_slope, atol=5e-2)
    approx(post(intercept)(0).mean, true_intercept, atol=5e-2)
