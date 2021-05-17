import numpy as np
from lab import B
from mlkernels import EQ, Delta, ZeroKernel

from stheno.model import Measure, GP
from ..util import approx


def test_summation_with_itself():
    p = GP(EQ())
    p_many = p + p + p + p + p

    x = B.linspace(0, 10, 5)
    approx(p_many(x).var, 25 * p(x).var)
    approx(p_many(x).mean, B.zeros(5, 1))

    y = B.randn(5, 1)
    post = p.measure | (p(x), y)
    approx(post(p_many)(x).mean, 5 * y)


def test_additive_model():
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


def test_fd_derivative():
    x = B.linspace(0, 10, 50)
    y = np.sin(x)

    p = GP(0.7 * EQ().stretch(1.0))
    dp = (p.shift(-1e-3) - p.shift(1e-3)) / 2e-3

    post = p.measure | (p(x), y)
    approx(post(dp)(x).mean, np.cos(x)[:, None], atol=1e-4)


def test_reflection():
    p = GP(EQ())
    p2 = 5 - p

    x = B.linspace(0, 5, 10)
    y = p(x).sample()

    post = p.measure | (p(x), y)
    approx(post(p2)(x).mean, 5 - y)

    post = p.measure | (p2(x), 5 - y)
    approx(post(p)(x).mean, y)


def test_negation():
    p = GP(EQ())
    p2 = -p

    x = B.linspace(0, 5, 10)
    y = p(x).sample()

    post = p.measure | (p(x), y)
    approx(post(p2)(x).mean, -y)

    post = p.measure | (p2(x), -y)
    approx(post(p)(x).mean, y)


def test_approximate_derivative():
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


def test_blr():
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
