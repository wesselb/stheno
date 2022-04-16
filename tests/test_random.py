import lab as B
import numpy as np
import pytest
from matrix import Dense, Zero
from plum import NotFoundLookupError
from scipy.stats import multivariate_normal

from stheno.random import Normal, RandomVector
from .util import approx


@pytest.fixture()
def normal1():
    mean = B.randn(3, 1)
    chol = B.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


@pytest.fixture()
def normal2():
    mean = B.randn(3, 1)
    chol = B.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


def test_normal_printing():
    dist = Normal(lambda: np.array([1, 2]), lambda: np.array([[3, 1], [1, 3]]))
    res = "<Normal:\n mean=unresolved,\n var=unresolved>"
    assert str(dist) == repr(dist) == res
    # Resolve mean.
    dist.mean
    assert str(dist) == "<Normal:\n mean=[1 2],\n var=unresolved>"
    assert repr(dist) == "<Normal:\n mean=array([1, 2]),\n var=unresolved>"
    # Resolve variance.
    dist.var
    assert str(dist) == (
        "<Normal:\n"
        " mean=[1 2],\n"
        " var=<dense matrix: batch=(), shape=(2, 2), dtype=int64>>"
    )
    assert repr(dist) == (
        "<Normal:\n"
        " mean=array([1, 2]),\n"
        " var=<dense matrix: batch=(), shape=(2, 2), dtype=int64\n"
        "      mat=[[3 1]\n"
        "           [1 3]]>>"
    )


def test_normal_mean_is_zero():
    # Check zero case.
    dist = Normal(B.eye(3))
    assert dist.mean_is_zero
    approx(dist.mean, B.zeros(3, 1))

    # Check another zero case.
    dist = Normal(Zero(np.float32, 3, 1), B.eye(3))
    assert dist.mean_is_zero
    approx(dist.mean, B.zeros(3, 1))

    # Check nonzero case.
    assert not Normal(B.randn(3, 1), B.eye(3)).mean_is_zero


def test_normal_lazy_zero_mean():
    dist = Normal(lambda: B.eye(3))

    # Querying `mean_is_zero` should not construct zeros.
    assert dist.mean_is_zero
    assert dist._mean is 0
    assert dist._var is None

    # Querying `mean` should now construct zeros.
    approx(dist.mean, B.zeros(3, 1))
    # At this point, the variance should be constructed, because it is used to get the
    # dimensionality and data type for the mean.
    assert dist._var is not None

    approx(dist.var, B.eye(3))


def test_normal_lazy_nonzero_mean():
    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3))
    # Nothing should be populated yet.
    assert dist._mean is None
    assert dist._var is None

    # But they should be populated upon request.
    approx(dist.mean, B.ones(3, 1))
    assert dist._var is None
    approx(dist.var, B.eye(3))


def test_normal_lazy_var_diag():
    # If `var_diag` isn't set, the variance will be constructed to get the diagonal.
    dist = Normal(lambda: B.eye(3))
    approx(dist.var_diag, B.ones(3))
    approx(dist._var, B.eye(3))

    # If `var_diag` is set, the variance will _not_ be constructed to get the diagonal.
    dist = Normal(lambda: B.eye(3), var_diag=lambda: 9)
    approx(dist.var_diag, 9)
    assert dist._var is None


def test_normal_lazy_mean_var():
    # The lazy `mean_var` should only be called when neither the mean nor the variance
    # exists. Otherwise, it's more efficient to just construct the other one. We
    # go over all branches in the `if`-statement.

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var=lambda: (8, 9))
    approx(dist.mean_var, (8, 9))
    approx(dist.mean, 8)
    approx(dist.var, 9)

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var=lambda: (8, 9))
    approx(dist.mean, B.ones(3, 1))
    approx(dist.mean_var, (B.ones(3, 1), B.eye(3)))
    approx(dist.var, B.eye(3))

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var=lambda: (8, 9))
    approx(dist.var, B.eye(3))
    approx(dist.mean_var, (B.ones(3, 1), B.eye(3)))
    approx(dist.mean, B.ones(3, 1))

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var=lambda: (8, 9))
    approx(dist.var, B.eye(3))
    approx(dist.mean, B.ones(3, 1))
    approx(dist.mean_var, (B.ones(3, 1), B.eye(3)))


def test_normal_lazy_mean_var_diag():
    # The lazy `mean_var_diag` should only be called when neither the mean nor the
    # diagonal of the variance exists. Otherwise, it's more efficient to just construct
    # the other one. We go over all branches in the `if`-statement.

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var_diag=lambda: (8, 9))
    approx(dist.marginals(), (8, 9))
    approx(dist.mean, 8)
    approx(dist.var_diag, 9)

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var_diag=lambda: (8, 9))
    approx(dist.mean, B.ones(3, 1))
    approx(dist.marginals(), (B.ones(3), B.ones(3)))
    approx(dist.var_diag, B.ones(3))

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var_diag=lambda: (8, 9))
    approx(dist.var_diag, B.ones(3))
    approx(dist.marginals(), (B.ones(3), B.ones(3)))
    approx(dist.mean, B.ones(3, 1))

    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3), mean_var_diag=lambda: (8, 9))
    approx(dist.var_diag, B.ones(3))
    approx(dist.mean, B.ones(3, 1))
    approx(dist.marginals(), (B.ones(3), B.ones(3)))


def test_normal_m2(normal1):
    approx(normal1.m2, normal1.var + normal1.mean @ normal1.mean.T)


def test_normal_marginals(normal1):
    mean, var = normal1.marginals()
    approx(mean, normal1.mean.squeeze())
    approx(var, B.diag(normal1.var))


def test_normal_marginal_credible_bounds(normal1):
    mean, lower, upper = normal1.marginal_credible_bounds()
    approx(mean, normal1.mean.squeeze())
    approx(lower, normal1.mean.squeeze() - 1.96 * B.diag(normal1.var) ** 0.5)
    approx(upper, normal1.mean.squeeze() + 1.96 * B.diag(normal1.var) ** 0.5)


def test_normal_diagonalise(normal1):
    approx(
        normal1.diagonalise(),
        Normal(normal1.mean, B.diag(B.diag(B.dense(normal1.var)))),
    )


def test_normal_logpdf(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    x = B.randn(3, 10)
    approx(normal1.logpdf(x), normal1_sp.logpdf(x.T), rtol=1e-6)

    # Test the the output of `logpdf` is flattened appropriately.
    assert B.shape(normal1.logpdf(B.ones(3, 1))) == ()
    assert B.shape(normal1.logpdf(B.ones(3, 2))) == (2,)


def test_normal_logpdf_missing_data(normal1):
    x = B.randn(3, 1)
    x[1] = B.nan
    approx(
        normal1.logpdf(x),
        Normal(
            normal1.mean[[0, 2]],
            normal1.var[[0, 2], :][:, [0, 2]],
        ).logpdf(x[[0, 2]]),
    )


def test_normal_entropy(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    approx(normal1.entropy(), normal1_sp.entropy())


def test_normal_kl(normal1, normal2):
    assert normal1.kl(normal1) < 1e-5
    assert normal1.kl(normal2) > 0.1

    # Test against Monte Carlo estimate.
    samples = normal1.sample(500_000)
    kl_est = B.mean(normal1.logpdf(samples)) - B.mean(normal2.logpdf(samples))
    kl = normal1.kl(normal2)
    approx(kl_est, kl, rtol=0.05)


def test_normal_w2(normal1, normal2):
    assert normal1.w2(normal1) < 5e-5
    assert normal1.w2(normal2) > 0.1


def test_normal_sampling():
    for mean in [0, 1]:
        dist = Normal(mean, 3 * B.eye(np.int32, 200))

        # Sample without noise.
        samples = dist.sample(2000)
        approx(B.mean(samples), mean, atol=5e-2)
        approx(B.std(samples) ** 2, 3, atol=5e-2)

        # Sample with noise
        samples = dist.sample(2000, noise=2)
        approx(B.mean(samples), mean, atol=5e-2)
        approx(B.std(samples) ** 2, 5, atol=5e-2)

        state, sample1 = dist.sample(B.create_random_state(B.dtype(dist), seed=0))
        state, sample2 = dist.sample(B.create_random_state(B.dtype(dist), seed=0))
        assert isinstance(state, B.RandomState)
        approx(sample1, sample2)


def test_normal_arithmetic(normal1, normal2):
    a = Dense(B.randn(3, 3))
    b = 5.0

    # Test matrix multiplication.
    approx(normal1.lmatmul(a).mean, a @ normal1.mean)
    approx(normal1.lmatmul(a).var, a @ normal1.var @ a.T)
    approx(normal1.rmatmul(a).mean, a.T @ normal1.mean)
    approx(normal1.rmatmul(a).var, a.T @ normal1.var @ a)

    # Test multiplication.
    approx((normal1 * b).mean, normal1.mean * b)
    approx((normal1 * b).var, normal1.var * b**2)
    approx((b * normal1).mean, normal1.mean * b)
    approx((b * normal1).var, normal1.var * b**2)
    with pytest.raises(NotFoundLookupError):
        normal1.__mul__(normal1)
    with pytest.raises(NotFoundLookupError):
        normal1.__rmul__(normal1)

    # Test addition.
    approx((normal1 + normal2).mean, normal1.mean + normal2.mean)
    approx((normal1 + normal2).var, normal1.var + normal2.var)
    approx(normal1.__radd__(b).mean, normal1.mean + b)
    approx(normal1.__radd__(b).mean, normal1.mean + b)
    with pytest.raises(NotFoundLookupError):
        normal1.__add__(RandomVector())
    with pytest.raises(NotFoundLookupError):
        normal1.__radd__(RandomVector())

    # Test negation.
    approx((-normal1).mean, -normal1.mean)
    approx((-normal1).var, normal1.var)

    # Test substraction.
    approx((normal1 - normal2).mean, normal1.mean - normal2.mean)
    approx((normal1 - normal2).var, normal1.var + normal2.var)
    approx(normal1.__rsub__(normal2).mean, normal2.mean - normal1.mean)
    approx(normal1.__rsub__(normal2).var, normal1.var + normal2.var)

    # Test division.
    approx(normal1.__div__(b).mean, normal1.mean / b)
    approx(normal1.__div__(b).var, normal1.var / b**2)
    approx(normal1.__truediv__(b).mean, normal1.mean / b)
    approx(normal1.__truediv__(b).var, normal1.var / b**2)


def test_normal_dtype(normal1):
    assert B.dtype(Normal(0, B.eye(3))) == np.float64
    assert B.dtype(Normal(B.ones(3), B.zeros(int, 3))) == np.float64
    assert B.dtype(Normal(B.ones(int, 3), B.zeros(int, 3))) == np.int64


def test_normal_cast(normal1):
    assert B.dtype(normal1) == np.float64
    assert B.dtype(B.cast(np.float32, normal1)) == np.float32
