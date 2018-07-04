# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
from lab import B

from stheno.graph import Graph, GP
from stheno.input import At
from stheno.kernel import Linear, EQ, Delta, Exp, RQ
from stheno.mean import TensorProductMean
from stheno.cache import Cache
# noinspection PyUnresolvedReferences,
from . import eq, raises, ok, le, eprint, lam


def abs_err(x1, x2=0): return np.sum(np.abs(x1 - x2))


def rel_err(x1, x2): return 2 * abs_err(x1, x2) / (abs_err(x1) + abs_err(x2))


def test_corner_cases():
    m1 = Graph()
    m2 = Graph()
    p1 = GP(EQ(), graph=m1)
    p2 = GP(EQ(), graph=m2)
    x = np.random.randn(10, 2)
    yield raises, RuntimeError, lambda: p1 + p2
    yield raises, NotImplementedError, lambda: p1 + p1(x)
    yield raises, NotImplementedError, lambda: p1 * p1(x)
    yield eq, str(GP(EQ(), graph=m1)), 'GP(EQ(), 0)'


def test_construction():
    p = GP(EQ(), graph=Graph())

    x = np.random.randn(10, 1)
    c = Cache()

    yield p.mean, x
    yield p.mean, At(p)(x)
    yield p.mean, x, c
    yield p.mean, At(p)(x), c

    yield p.kernel, x
    yield p.kernel, At(p)(x)
    yield p.kernel, x, c
    yield p.kernel, At(p)(x), c
    yield p.kernel, x, x
    yield p.kernel, At(p)(x), x
    yield p.kernel, x, At(p)(x)
    yield p.kernel, At(p)(x), At(p)(x)
    yield p.kernel, x, x, c
    yield p.kernel, At(p)(x), x, c
    yield p.kernel, x, At(p)(x), c
    yield p.kernel, At(p)(x), At(p)(x), c

    yield p.kernel.elwise, x
    yield p.kernel.elwise, At(p)(x)
    yield p.kernel.elwise, x, c
    yield p.kernel.elwise, At(p)(x), c
    yield p.kernel.elwise, x, x
    yield p.kernel.elwise, At(p)(x), x
    yield p.kernel.elwise, x, At(p)(x)
    yield p.kernel.elwise, At(p)(x), At(p)(x)
    yield p.kernel.elwise, x, x, c
    yield p.kernel.elwise, At(p)(x), x, c
    yield p.kernel.elwise, x, At(p)(x), c
    yield p.kernel.elwise, At(p)(x), At(p)(x), c


def test_sum_other():
    model = Graph()
    p1 = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    p2 = p1 + 5.
    p3 = 5. + p1
    p4 = model.sum(5., p1)

    x = np.random.randn(5, 1)
    yield assert_allclose, p1.mean(x) + 5., p2.mean(x)
    yield assert_allclose, p1.mean(x) + 5., p3.mean(x)
    yield assert_allclose, p1.mean(x) + 5., p4.mean(x)
    yield assert_allclose, p1.kernel(x), p2.kernel(x)
    yield assert_allclose, p1.kernel(x), p3.kernel(x)
    yield assert_allclose, p1.kernel(x), p4.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p3)(x)), \
          p1.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p4)(x)), \
          p1.kernel(x)


def test_mul_other():
    model = Graph()
    p1 = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    p2 = 5. * p1
    p3 = p1 * 5.

    x = np.random.randn(5, 1)
    yield assert_allclose, 5. * p1.mean(x), p2.mean(x)
    yield assert_allclose, 5. * p1.mean(x), p3.mean(x)
    yield assert_allclose, 25. * p1.kernel(x), p2.kernel(x)
    yield assert_allclose, 25. * p1.kernel(x), p3.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p3)(x)), \
          25. * p1.kernel(x)


def test_at_shorthand():
    model = Graph()
    p1 = GP(EQ(), graph=model)
    x = p1.__matmul__(1)

    yield eq, type(x), At(p1)
    yield eq, x.get(), 1


def test_properties():
    model = Graph()

    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ().stretch(2), graph=model)
    p3 = GP(EQ().periodic(10), graph=model)

    p = p1 + 2 * p2

    yield eq, p.stationary, True, 'stationary'
    yield eq, p.var, 1 + 2 ** 2, 'var'
    yield assert_allclose, p.length_scale, \
          (1 + 2 * 2 ** 2) / (1 + 2 ** 2)
    yield eq, p.period, np.inf, 'period'

    yield eq, p3.period, 10, 'period'

    p = p3 + p

    yield eq, p.stationary, True, 'stationary 2'
    yield eq, p.var, 1 + 2 ** 2 + 1, 'var 2'
    yield eq, p.period, np.inf, 'period 2'

    p = p + GP(Linear(), graph=model)

    yield eq, p.stationary, False, 'stationary 3'


def test_terms_factors():
    model = Graph()
    p = 2 * GP(EQ() * RQ(1) + Linear(), lambda x: x, graph=model) + 3

    yield eq, p.kernel.num_factors, 2
    yield eq, str(p.kernel.factor(0)), '4'
    yield eq, str(p.kernel.factor(1)), 'EQ() * RQ(1) + Linear()'
    yield eq, p.kernel.factor(1).num_terms, 2
    yield eq, str(p.kernel.factor(1).term(0)), 'EQ() * RQ(1)'
    yield eq, str(p.kernel.factor(1).term(1)), 'Linear()'

    yield eq, p.mean.num_terms, 2
    yield eq, str(p.mean.term(0)), '2 * <lambda>'
    yield eq, str(p.mean.term(1)), '3 * 1'
    yield eq, p.mean.term(0).num_factors, 2
    yield eq, str(p.mean.term(0).factor(0)), '2'
    yield eq, str(p.mean.term(0).factor(1)), '<lambda>'


def test_case_summation_with_itself():
    # Test summing the same GP with itself.
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = p1 + p1 + p1 + p1 + p1
    x = np.linspace(0, 10, 5)[:, None]

    yield assert_allclose, p2(x).var, 25 * p1(x).var
    yield assert_allclose, p2(x).mean, np.zeros((5, 1))

    y = np.random.randn(5, 1)
    model.condition(At(p1)(x), y)

    yield assert_allclose, p2(x).mean, 5 * y


def test_case_additive_model():
    # Test some additive model.

    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ(), graph=model)
    p3 = p1 + p2

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y1 = p1(x).sample()
    y2 = p2(x).sample()

    # First, test independence:
    yield assert_allclose, p1.kernel(At(p2)(x), x), np.zeros((n, n))
    yield assert_allclose, p1.kernel(At(p2)(x), At(p1)(x)), np.zeros((n, n))
    yield assert_allclose, p1.kernel(x, At(p2)(x)), np.zeros((n, n))
    yield assert_allclose, p1.kernel(At(p1)(x), At(p2)(x)), np.zeros((n, n))

    # Now run through some test cases:

    model.condition(At(p1)(x), y1)
    model.condition(At(p2)(x), y2)
    yield assert_allclose, p3(x).mean, y1 + y2
    model.revert_prior()

    model.condition(At(p2)(x), y2)
    model.condition(At(p1)(x), y1)
    yield assert_allclose, p3(x).mean, y1 + y2
    model.revert_prior()

    model.condition(At(p1)(x), y1)
    model.condition(At(p3)(x), y1 + y2)
    yield assert_allclose, p2(x).mean, y2
    model.revert_prior()

    model.condition(At(p3)(x), y1 + y2)
    model.condition(At(p1)(x), y1)
    yield assert_allclose, p2(x).mean, y2
    model.revert_prior()

    yield assert_allclose, p3 \
        .condition(At(p1)(x), y1) \
        .condition(At(p2)(x), y2)(x) \
        .mean, y1 + y2
    p3.revert_prior()

    yield assert_allclose, p3 \
        .condition(At(p2)(x), y2) \
        .condition(At(p1)(x), y1)(x) \
        .mean, y1 + y2
    p3.revert_prior()

    yield assert_allclose, p2 \
        .condition(At(p3)(x), y1 + y2) \
        .condition(At(p1)(x), y1)(x) \
        .mean, y2
    p3.revert_prior()

    yield assert_allclose, p2 \
        .condition(At(p1)(x), y1) \
        .condition(At(p3)(x), y1 + y2)(x) \
        .mean, y2
    p3.revert_prior()

    yield assert_allclose, p3.condition(x, y1 + y2)(x).mean, y1 + y2


def test_shifting():
    model = Graph()

    p = GP(EQ(), graph=model)
    p2 = p.shift(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x), y)(x - 5).mean, y
    yield le, abs_err(p(x - 5).spd.diag), 1e-10
    p.revert_prior()
    yield assert_allclose, p2.condition(At(p)(x), y)(x + 5).mean, y
    yield le, abs_err(p2(x + 5).spd.diag), 1e-10
    p.revert_prior()


def test_stretching():
    model = Graph()

    p = GP(EQ(), graph=model)
    p2 = p.stretch(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x), y)(x / 5).mean, y
    yield le, abs_err(p(x / 5).spd.diag), 1e-10
    p.revert_prior()
    yield assert_allclose, p2.condition(At(p)(x), y)(x * 5).mean, y
    yield le, abs_err(p2(x * 5).spd.diag), 1e-10
    p.revert_prior()


def test_input_transform():
    model = Graph()

    p = GP(EQ(), graph=model)
    p2 = p.transform(lambda x, B: x / 5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x), y)(x / 5).mean, y
    yield le, abs_err(p(x / 5).spd.diag), 1e-10
    p.revert_prior()
    yield assert_allclose, p2.condition(At(p)(x), y)(x * 5).mean, y
    yield le, abs_err(p2(x * 5).spd.diag), 1e-10
    p.revert_prior()


def test_selection():
    model = Graph()

    p = GP(EQ(), graph=model)  # 1D
    p2 = p.select(0)  # 2D

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    x1 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    x2 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x1), y)(x).mean, y
    yield le, abs_err(p(x).spd.diag), 1e-10
    p.revert_prior()

    yield assert_allclose, p.condition(At(p2)(x2), y)(x).mean, y
    yield le, abs_err(p(x).spd.diag), 1e-10
    p.revert_prior()

    yield assert_allclose, p2.condition(At(p)(x), y)(x1).mean, y
    yield assert_allclose, p2(x2).mean, y
    yield le, abs_err(p2(x1).spd.diag), 1e-10
    yield le, abs_err(p2(x2).spd.diag), 1e-10
    p.revert_prior()


def test_case_fd_derivative():
    x = np.linspace(0, 10, 50)[:, None]
    y = np.sin(x)

    model = Graph()
    p = GP(.7 * EQ().stretch(1.), graph=model)
    dp = (p.shift(-1e-3) - p.shift(1e-3)) / 2e-3

    yield le, abs_err(np.cos(x) - dp.condition(At(p)(x), y)(x).mean), 1e-4


def test_case_reflection():
    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = 5 - p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    model.condition(At(p)(x), y)
    yield le, abs_err(p2(x).mean - (5 - y)), 1e-5
    model.revert_prior()
    model.condition(At(p2)(x), 5 - y)
    yield le, abs_err(p(x).mean - y), 1e-5
    model.revert_prior()

    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = -p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    model.condition(At(p)(x), y)
    yield le, abs_err(p2(x).mean + y), 1e-5
    model.revert_prior()
    model.condition(At(p2)(x), -y)
    yield le, abs_err(p(x).mean - y), 1e-5
    model.revert_prior()


def test_case_exact_derivative():
    B.backend_to_tf()
    s = B.Session()

    model = Graph()
    x = np.linspace(0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ(), graph=model)
    dp = p.diff()

    # Test conditioning on function.
    p.condition(x, y)
    yield le, abs_err(s.run(dp(x).mean - 2)), 1e-3
    p.revert_prior()

    # Test conditioning on derivative.
    dp.condition(x, y)
    p.condition(np.zeros((1, 1)), np.zeros((1, 1)))  # Fix integration constant.
    yield le, abs_err(s.run(p(x).mean - x ** 2)), 1e-3
    p.revert_prior()

    s.close()
    B.backend_to_np()


def test_case_approximate_derivative():
    model = Graph()
    x = np.linspace(0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ().stretch(1.), graph=model)
    dp = p.diff_approx()

    # Test conditioning on function.
    p.condition(x, y)
    yield le, abs_err(dp(x).mean - 2), 1e-3
    p.revert_prior()

    # Add some regularisation for this test case.
    orig_epsilon = B.epsilon
    B.epsilon = 1e-10

    # Test conditioning on derivative.
    dp.condition(x, y)
    p.condition(0, 0)  # Fix integration constant.
    yield le, abs_err(p(x).mean - x ** 2), 1e-3
    p.revert_prior()

    # Set regularisation back.
    B.epsilon = orig_epsilon


def test_case_blr():
    model = Graph()
    x = np.linspace(0, 10, 100)

    slope = GP(1, graph=model)
    intercept = GP(1, graph=model)
    f = slope * (lambda x: x) + intercept
    y = f + 1e-2 * GP(Delta(), graph=model)

    # Sample true slope and intercept.
    true_slope = slope(0).sample()
    true_intercept = intercept.condition(At(slope)(0), true_slope)(0).sample()

    # Sample observations.
    y_obs = y.condition(At(intercept)(0), true_intercept)(x).sample()
    model.revert_prior()

    # Predict.
    model.condition(At(y)(x), y_obs)
    mean_slope, mean_intercept = slope(0).mean, intercept(0).mean
    model.revert_prior()

    yield le, np.abs(true_slope[0, 0] - mean_slope[0, 0]), 5e-2
    yield le, np.abs(true_intercept[0, 0] - mean_intercept[0, 0]), 5e-2


def test_multi_sample():
    model = Graph()
    p1 = GP(0, 1, graph=model)
    p2 = GP(0, 2, graph=model)
    p3 = GP(0, 3, graph=model)

    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 20)
    x3 = np.linspace(0, 1, 30)

    s1, s2, s3 = model.sample(At(p1)(x1), At(p2)(x2), At(p3)(x3))

    yield eq, s1.shape, (10, 1)
    yield eq, s2.shape, (20, 1)
    yield eq, s3.shape, (30, 1)
    yield eq, np.shape(model.sample(At(p1)(x1))), (10, 1)

    yield le, abs_err(s1 - 1), 1e-4
    yield le, abs_err(s2 - 2), 1e-4
    yield le, abs_err(s3 - 3), 1e-4


def test_multi_conditioning():
    model = Graph()

    p1 = GP(EQ(), graph=model)
    p2 = GP(2 * Exp().stretch(2), graph=model)
    p3 = GP(.5 * RQ(1e-1).stretch(.5), graph=model)

    p = p1 + p2 + p3

    x1 = np.linspace(0, 2, 10)
    x2 = np.linspace(1, 3, 10)
    x3 = np.linspace(0, 3, 10)

    s1, s2 = model.sample(At(p1)(x1), At(p2)(x2))

    post1 = p.condition(At(p1)(x1), s1).condition(At(p2)(x2), s2)(x3)
    model.revert_prior()

    post2 = p.condition((At(p1)(x1), s1), (At(p2)(x2), s2))(x3)
    model.revert_prior()

    post3 = p.condition((At(p2)(x2), s2), (At(p1)(x1), s1))(x3)
    model.revert_prior()

    p2.condition((x2, s2), (At(p1)(x1), s1))
    post4 = p(x3)
    model.revert_prior()

    yield assert_allclose, post1.mean, post2.mean
    yield assert_allclose, post1.mean, post3.mean
    yield assert_allclose, post1.mean, post4.mean
    yield assert_allclose, post1.var, post2.var
    yield assert_allclose, post1.var, post3.var
    yield assert_allclose, post1.var, post4.var

    # Test `At` check.
    yield raises, ValueError, lambda: model.condition((0, 0))


def test_approximate_multiplication():
    model = Graph()

    # Construct model.
    p1 = GP(EQ(), 20, graph=model)
    p2 = GP(EQ(), 20, graph=model)
    p_prod = p1 * p2
    x = np.linspace(0, 10, 50)

    # Sample functions.
    s1, s2 = model.sample(At(p1)(x), At(p2)(x))

    # Infer product.
    model.condition((At(p1)(x), s1), (At(p2)(x), s2))
    yield le, rel_err(p_prod(x).mean, s1 * s2), 1e-2
    model.revert_prior()

    # Perform division.
    cur_epsilon = B.epsilon
    B.epsilon = 1e-8
    model.condition((At(p1)(x), s1), (At(p_prod)(x), s1 * s2))
    yield le, rel_err(p2(x).mean, s2), 1e-2
    model.revert_prior()
    B.epsilon = cur_epsilon
