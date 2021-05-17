import pytest
import tensorflow as tf
from lab import B
from matrix import Diagonal
from mlkernels import (
    Linear,
    EQ,
    Delta,
    Exp,
    TensorProductMean,
)

from stheno.model import Measure, GP, Obs, PseudoObs, cross, FDD
from .util import assert_equal_normals, assert_equal_measures
from ..util import approx


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


def test_default_measure():
    with Measure() as m1:
        p1 = GP(EQ())

        with Measure() as m2:
            p2 = GP(EQ())

        p3 = GP(EQ())

    p4 = GP(EQ())

    assert p1.measure is m1
    assert p2.measure is m2
    assert p3.measure is m1
    assert p4.measure is not m1
    assert p4.measure is not m2


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


@pytest.mark.parametrize(
    "generate_noise_tuple",
    [
        lambda x: (),
        lambda x: (B.rand(),),
        lambda x: (B.rand(x),),
        lambda x: (B.diag(B.rand(x)),),
        lambda x: (Diagonal(B.rand(x)),),
    ],
)
def test_conditioning(generate_noise_tuple):
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
    p_sum = p1 + p2

    # Sample some data to condition on.
    x1 = B.linspace(0, 2, 3)
    n1 = generate_noise_tuple(x1)
    y1 = p1(x1, *n1).sample()
    tup1 = (p1(x1, *n1), y1)
    x_sum = B.linspace(3, 5, 3)
    n_sum = generate_noise_tuple(x_sum)
    y_sum = p_sum(x_sum, *n_sum).sample()
    tup_sum = (p_sum(x_sum, *n_sum), y_sum)

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
        m.condition(*tup_sum),
        m.condition(tup_sum),
        m | tup_sum,
        m | (tup_sum,),
        m | Obs(*tup_sum),
        m | Obs(tup_sum),
    )

    assert_equal_measures(
        fdds_check,
        m.condition(tup1, tup_sum),
        m | (tup1, tup_sum),
        m | Obs(tup1, tup_sum),
    )

    # Check that conditioning gives an FDD and that it is consistent.
    post = m | tup1
    assert isinstance(post(p1(x1, 0.1)), FDD)
    assert_equal_measures(post(p1(x1, 0.1)), post(p1)(x1, 0.1))


def test_conditioning_consistency():
    m = Measure()
    p = GP(EQ(), measure=m)
    e = GP(0.1 * Delta(), measure=m)
    e2 = GP(e.kernel, measure=m)

    x = B.linspace(0, 5, 10)
    y = (p + e)(x).sample()

    post1 = m | ((p + e)(x), y)
    post2 = m | (p(x, 0.1), y)

    assert_equal_measures([p(x), (p + e2)(x)], post1, post2)
    with pytest.raises(AssertionError):
        assert_equal_normals(post1((p + e)(x)), post2((p + e)(x)))


def test_conditioning_prior():
    p = GP(EQ())
    x = B.zeros(0, 1)
    y = B.zeros(0, 1)
    post = p.measure | (p(x), y)
    assert post(p).mean is p.mean
    assert post(p).kernel is p.kernel


def test_conditioning_shorthand():
    p = GP(EQ())

    # Test conditioning once.
    x = B.linspace(0, 5, 10)
    y = p(x).sample()
    p_post1 = p.condition(p(x), y)
    p_post2 = p | (p(x), y)
    approx(p_post1.mean(x), y)
    approx(p_post2.mean(x), y)

    # Test conditioning twice.
    x = B.linspace(10, 20, 10)
    y = p(x).sample()
    p_post1 = p_post1.condition(p(x), y)
    p_post2 = p_post2 | (p(x), y)
    approx(p_post1.mean(x), y)
    approx(p_post2.mean(x), y)


@pytest.mark.parametrize(
    "generate_noise_tuple",
    [
        lambda x: (B.rand(),),
        lambda x: (B.rand(x),),
        lambda x: (Diagonal(B.rand(x)),),
    ],
)
def test_pseudo_conditioning_and_elbo(generate_noise_tuple):
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
    p_sum = p1 + p2

    # Sample some data to condition on.
    x1 = B.linspace(0, 2, 3)
    n1 = generate_noise_tuple(x1)
    y1 = p1(x1, *n1).sample()
    tup1 = (p1(x1, *n1), y1)
    x_sum = B.linspace(3, 5, 3)
    n_sum = generate_noise_tuple(x_sum)
    y_sum = p_sum(x_sum, *n_sum).sample()
    tup_sum = (p_sum(x_sum, *n_sum), y_sum)

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
        m | tup_sum,
        m | PseudoObs(p_sum(x_sum), *tup_sum),
        m | PseudoObs((p_sum(x_sum),), *tup_sum),
        m | PseudoObs((p_sum(x_sum), p1(x1)), *tup_sum),
        m | PseudoObs(p_sum(x_sum), tup_sum),
        m | PseudoObs((p_sum(x_sum),), tup_sum),
        m.condition(PseudoObs((p_sum(x_sum), p1(x1)), tup_sum)),
    )
    approx(
        m.logpdf(Obs(*tup_sum)),
        PseudoObs(p_sum(x_sum), tup_sum).elbo(m),
    )

    # Check conditioning and ELBO on two data sets.
    assert_equal_measures(
        fdds_check,
        m | (tup_sum, tup1),
        m.condition(PseudoObs((p_sum(x_sum), p1(x1)), tup_sum, tup1)),
    )
    approx(
        m.logpdf(Obs(tup_sum, tup1)),
        PseudoObs((p_sum(x_sum), p1(x1)), tup_sum, tup1).elbo(m),
    )

    # The following lose information, so check them separately.
    assert_equal_measures(
        fdds_check,
        m | PseudoObs(p_sum(x_sum), tup_sum, tup1),
        m | PseudoObs((p_sum(x_sum),), tup_sum, tup1),
    )

    # Test caching.
    for name in ["K_z", "elbo", "mu", "A"]:
        obs = PseudoObs(p_sum(x_sum), *tup_sum)
        assert getattr(obs, name)(m) is getattr(obs, name)(m)

    # Test requirement that noise must be diagonal.
    with pytest.raises(RuntimeError):
        PseudoObs(p_sum(x_sum), (p_sum(x_sum, p_sum(x_sum).var), y_sum)).elbo(m)

    # Test that noise on inducing points loses information.
    with pytest.raises(AssertionError):
        assert_equal_measures(
            fdds_check,
            m | tup_sum,
            m | PseudoObs(p_sum(x_sum, 0.1), *tup_sum),
        )


def test_backward_compatibility():
    from stheno import SparseObs, SparseObservations

    assert SparseObs is PseudoObs
    assert SparseObservations is PseudoObs


def test_logpdf():
    m = Measure()
    p1 = GP(EQ(), measure=m)
    p2 = GP(Exp(), measure=m)
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

    # Check that `Measure.logpdf` allows `Obs` and `PseudoObs`.
    obs = Obs(p3(x3), y3)
    approx(m.logpdf(obs), p3(x3).logpdf(y3))
    obs = PseudoObs(p3(x3), p3(x3, 1), y3)
    approx(m.logpdf(obs), p3(x3, 1).logpdf(y3))


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
