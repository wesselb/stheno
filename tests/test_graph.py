# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B
from plum import type_parameter

from stheno.cache import Cache
from stheno.graph import Graph, GP, Obs, SparseObs
from stheno.input import At, Unique
from stheno.kernel import Linear, EQ, Delta, Exp, RQ, ZeroKernel, OneKernel, \
    ScaledKernel
from stheno.mean import TensorProductMean, ZeroMean, ScaledMean, OneMean
from stheno.random import Normal
# noinspection PyUnresolvedReferences,
from . import eq, raises, ok, le, assert_allclose, assert_instance, eprint


def abs_err(x1, x2=0):
    return np.sum(np.abs(x1 - x2))


def rel_err(x1, x2):
    return 2 * abs_err(x1, x2) / (abs_err(x1) + abs_err(x2))


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
    model = Graph()
    p = GP(EQ(), graph=model)

    x = np.random.randn(10, 1)
    c = Cache()

    yield p.mean, x
    yield p.mean, p(x)
    yield p.mean, x, c
    yield p.mean, p(x), c

    yield p.kernel, x
    yield p.kernel, p(x)
    yield p.kernel, x, c
    yield p.kernel, p(x), c
    yield p.kernel, x, x
    yield p.kernel, p(x), x
    yield p.kernel, x, p(x)
    yield p.kernel, p(x), p(x)
    yield p.kernel, x, x, c
    yield p.kernel, p(x), x, c
    yield p.kernel, x, p(x), c
    yield p.kernel, p(x), p(x), c

    yield p.kernel.elwise, x
    yield p.kernel.elwise, p(x)
    yield p.kernel.elwise, x, c
    yield p.kernel.elwise, p(x), c
    yield p.kernel.elwise, x, x
    yield p.kernel.elwise, p(x), x
    yield p.kernel.elwise, x, p(x)
    yield p.kernel.elwise, p(x), p(x)
    yield p.kernel.elwise, x, x, c
    yield p.kernel.elwise, p(x), x, c
    yield p.kernel.elwise, x, p(x), c
    yield p.kernel.elwise, p(x), p(x), c

    # Test resolution of kernel and mean.
    k = EQ()
    m = TensorProductMean(lambda x: x ** 2)

    yield assert_instance, GP(k, graph=model).mean, ZeroMean
    yield assert_instance, GP(k, 5, graph=model).mean, ScaledMean
    yield assert_instance, GP(k, 1, graph=model).mean, OneMean
    yield assert_instance, GP(k, 0, graph=model).mean, ZeroMean
    yield assert_instance, GP(k, m, graph=model).mean, TensorProductMean
    yield assert_instance, GP(k, graph=model).kernel, EQ
    yield assert_instance, GP(5, graph=model).kernel, ScaledKernel
    yield assert_instance, GP(1, graph=model).kernel, OneKernel
    yield assert_instance, GP(0, graph=model).kernel, ZeroKernel

    # Test construction of finite-dimensional distribution.
    d = GP(k, m, graph=model)(x)
    yield assert_allclose, d.var, k(x)
    yield assert_allclose, d.mean, m(x)


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
    yield assert_allclose, p1.kernel(p2(x), p3(x)), \
          p1.kernel(x)
    yield assert_allclose, p1.kernel(p2(x), p4(x)), \
          p1.kernel(x)

    # Check that a `GP` cannot be summed with a `Normal`.
    yield raises, NotImplementedError, lambda: p1 + Normal(np.eye(3))
    yield raises, NotImplementedError, lambda: Normal(np.eye(3)) + p1


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
    yield assert_allclose, model.kernels[p2, p3](x, x), 25. * p1.kernel(x)

    # Check that a `GP` cannot be multiplied with a `Normal`.
    yield raises, NotImplementedError, lambda: p1 * Normal(np.eye(3))
    yield raises, NotImplementedError, lambda: Normal(np.eye(3)) * p1


def test_shorthands():
    model = Graph()
    p = GP(EQ(), graph=model)

    # Construct a normal distribution that serves as in input.
    x = p(1)
    yield assert_instance, x, At
    yield ok, type_parameter(x) is p
    yield eq, x.get(), 1
    yield eq, str(p(x)), '{}({})'.format(str(p), str(x))
    yield eq, repr(p(x)), '{}({})'.format(repr(p), repr(x))

    # Construct a normal distribution that does not serve as an input.
    x = Normal(np.ones((1, 1)))
    yield raises, RuntimeError, lambda: type_parameter(x)
    yield raises, RuntimeError, lambda: x.get()
    yield raises, RuntimeError, lambda: p | (x, 1)

    # Test shorthands for stretching and selection.
    p = GP(EQ(), graph=Graph())
    yield eq, str(p > 2), str(p.stretch(2))
    yield eq, str(p[0]), str(p.select(0))


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


def test_predict():
    model = Graph()
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    x = np.linspace(0, 5, 10)

    # Check that `predict` outputs the right thing.
    mean, lower, upper = p.predict(x)
    var = B.diag(p.kernel(x))
    yield assert_allclose, mean, p.mean(x)[:, 0]
    yield assert_allclose, lower, p.mean(x)[:, 0] - 2 * var ** .5
    yield assert_allclose, upper, p.mean(x)[:, 0] + 2 * var ** .5

    # Test correctness.
    y = p(x).sample()
    mean, lower, upper = (p | (x, y)).predict(x)
    yield assert_allclose, mean, y[:, 0]
    yield le, B.mean(B.abs(upper - lower)), 1e-5

    mean, lower, upper = (p | (x, y)).predict(x + 100)
    yield assert_allclose, mean, p.mean(x + 100)[:, 0]
    yield assert_allclose, upper - lower, 4 * np.ones(10)


def test_observations_and_conditioning():
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ(), graph=model)
    p = p1 + p2
    x = np.linspace(0, 5, 10)
    y = p(x).sample()
    y1 = p1(x).sample()

    # Test all ways of conditioning, including shorthands.
    obs1 = Obs(p(x), y)
    obs2 = Obs(x, y, ref=p)
    yield assert_allclose, obs1.y, obs2.y
    yield assert_allclose, obs1.K_x, obs2.K_x

    obs3 = Obs((p(x), y), (p1(x), y1))
    obs4 = Obs((x, y), (p1(x), y1), ref=p)
    yield assert_allclose, obs3.y, obs4.y
    yield assert_allclose, obs3.K_x, obs4.K_x

    def assert_equal_mean_var(x, *ys):
        for y in ys:
            yield assert_allclose, x.mean, y.mean
            yield assert_allclose, x.var, y.var

    for test in assert_equal_mean_var(p.condition(x, y)(x),
                                      p.condition(p(x), y)(x),
                                      (p | (x, y))(x),
                                      (p | (p(x), y))(x),
                                      p.condition(obs1)(x),
                                      p.condition(obs2)(x),
                                      (p | obs1)(x),
                                      (p | obs2)(x)):
        yield test

    for test in assert_equal_mean_var(p.condition((x, y), (p1(x), y1))(x),
                                      p.condition((p(x), y), (p1(x), y1))(x),
                                      (p | [(x, y), (p1(x), y1)])(x),
                                      (p | [(p(x), y), (p1(x), y1)])(x),
                                      p.condition(obs3)(x),
                                      p.condition(obs4)(x),
                                      (p | obs3)(x),
                                      (p | obs4)(x)):
        yield test

    # Check conditioning multiple processes at once.
    p1_post, p2_post, p_post = (p1, p2, p) | obs1
    p1_post, p2_post, p_post = p1_post(x), p2_post(x), p_post(x)
    p1_post2, p2_post2, p_post2 = (p1 | obs1)(x), (p2 | obs1)(x), (p | obs1)(x)

    yield assert_allclose, p1_post.mean, p1_post2.mean
    yield assert_allclose, p1_post.var, p1_post2.var
    yield assert_allclose, p2_post.mean, p2_post2.mean
    yield assert_allclose, p2_post.var, p2_post2.var
    yield assert_allclose, p_post.mean, p_post2.mean
    yield assert_allclose, p_post.var, p_post2.var

    # Test `At` check.
    yield raises, ValueError, lambda: Obs(0, 0)
    yield raises, ValueError, lambda: Obs((0, 0), (0, 0))
    yield raises, ValueError, lambda: SparseObs(0, p, (0, 0))


def test_stretching():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    yield eq, str(p.stretch(1)), 'GP(EQ() > 1, <lambda> > 1)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.stretch(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    yield assert_allclose, post(x / 5).mean, y
    yield le, abs_err(B.diag(post(x / 5).var)), 1e-10

    post = p2.condition(p(x), y)
    yield assert_allclose, post(x * 5).mean, y
    yield le, abs_err(B.diag(post(x * 5).var)), 1e-10


def test_shifting():
    model = Graph()

    # Test construction:
    p = GP(Linear(), TensorProductMean(lambda x: x ** 2), graph=model)
    yield eq, str(p.shift(1)), 'GP(Linear() shift 1, <lambda> shift 1)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.shift(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    yield assert_allclose, post(x - 5).mean, y
    yield le, abs_err(B.diag(post(x - 5).var)), 1e-10

    post = p2.condition(p(x), y)
    yield assert_allclose, post(x + 5).mean, y
    yield le, abs_err(B.diag(post(x + 5).var)), 1e-10


def test_input_transform():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    yield eq, str(p.transform(lambda x, c: x)), \
          'GP(EQ() transform <lambda>, <lambda> transform <lambda>)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.transform(lambda x, B: x / 5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    yield assert_allclose, post(x / 5).mean, y
    yield le, abs_err(B.diag(post(x / 5).var)), 1e-10

    post = p2.condition(p(x), y)
    yield assert_allclose, post(x * 5).mean, y
    yield le, abs_err(B.diag(post(x * 5).var)), 1e-10


def test_selection():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    yield eq, str(p.select(1)), 'GP(EQ() : [1], <lambda> : [1])'
    yield eq, str(p.select(1, 2)), 'GP(EQ() : [1, 2], <lambda> : [1, 2])'

    # Test case:
    p = GP(EQ(), graph=model)  # 1D
    p2 = p.select(0)  # 2D

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    x1 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    x2 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    y = p2(x).sample()

    post = p.condition(p2(x1), y)
    yield assert_allclose, post(x).mean, y
    yield le, abs_err(B.diag(post(x).var)), 1e-10

    post = p.condition(p2(x2), y)
    yield assert_allclose, post(x).mean, y
    yield le, abs_err(B.diag(post(x).var)), 1e-10

    post = p2.condition(p(x), y)
    yield assert_allclose, post(x1).mean, y
    yield assert_allclose, post(x2).mean, y
    yield le, abs_err(B.diag(post(x1).var)), 1e-10
    yield le, abs_err(B.diag(post(x2).var)), 1e-10


def test_derivative():
    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=Graph())
    yield eq, str(p.diff(1)), 'GP(d(1) EQ(), d(1) <lambda>)'

    # Test case:
    B.backend_to_tf()
    s = B.Session()

    model = Graph()
    x = np.linspace(0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ(), graph=model)
    dp = p.diff()

    # Test conditioning on function.
    yield le, abs_err(s.run(dp.condition(p(x), y)(x).mean - 2)), 1e-3

    # Test conditioning on derivative.
    post = p.condition((B.cast(0., np.float64), B.cast(0., np.float64)),
                       (dp(x), y))
    yield le, abs_err(s.run(post(x).mean - x ** 2)), 1e-3

    s.close()
    B.backend_to_np()


def test_multi_sample():
    model = Graph()
    p1 = GP(0, 1, graph=model)
    p2 = GP(0, 2, graph=model)
    p3 = GP(0, 3, graph=model)

    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 20)
    x3 = np.linspace(0, 1, 30)

    s1, s2, s3 = model.sample(p1(x1), p2(x2), p3(x3))

    yield eq, s1.shape, (10, 1)
    yield eq, s2.shape, (20, 1)
    yield eq, s3.shape, (30, 1)
    yield eq, np.shape(model.sample(p1(x1))), (10, 1)

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

    s1, s2 = model.sample(p1(x1), p2(x2))

    post1 = ((p | (p1(x1), s1)) | ((p2 | (p1(x1), s1))(x2), s2))(x3)
    post2 = (p | ((p1(x1), s1), (p2(x2), s2)))(x3)
    post3 = (p | ((p2(x2), s2), (p1(x1), s1)))(x3)

    yield assert_allclose, post1.mean, post2.mean, 'means 1', 1e-6, 1e-6
    yield assert_allclose, post1.mean, post3.mean, 'means 2', 1e-6, 1e-6
    yield assert_allclose, post1.var, post2.var
    yield assert_allclose, post1.var, post3.var


def test_approximate_multiplication():
    model = Graph()

    # Construct model.
    p1 = GP(EQ(), 20, graph=model)
    p2 = GP(EQ(), 20, graph=model)
    p_prod = p1 * p2
    x = np.linspace(0, 10, 50)

    # Sample functions.
    s1, s2 = model.sample(p1(x), p2(x))

    # Infer product.
    post = p_prod.condition((p1(x), s1), (p2(x), s2))
    yield le, rel_err(post(x).mean, s1 * s2), 1e-2

    # Perform division.
    cur_epsilon = B.epsilon
    B.epsilon = 1e-8
    post = p2.condition((p1(x), s1), (p_prod(x), s1 * s2))
    yield le, rel_err(post(x).mean, s2), 1e-2
    B.epsilon = cur_epsilon

    # Check graph check.
    model2 = Graph()
    p3 = GP(EQ(), graph=model2)
    yield raises, RuntimeError, lambda: p3 * p1


def test_naming():
    model = Graph()

    p1 = GP(EQ(), 1, graph=model)
    p2 = GP(EQ(), 2, graph=model)

    # Test setting and getting names.
    p1.name = 'name'

    yield ok, model['name'] is p1
    yield eq, p1.name, 'name'
    yield eq, model[p1], 'name'
    yield raises, KeyError, lambda: model['other_name']
    yield raises, KeyError, lambda: model[p2]

    # Check that names can not be doubly assigned.
    def doubly_assign():
        p2.name = 'name'

    yield raises, RuntimeError, doubly_assign

    # Move name to other GP.
    p1.name = 'other_name'
    p2.name = 'name'

    # Check that everything has been properly assigned.
    yield ok, model['name'] is p2
    yield eq, p2.name, 'name'
    yield eq, model[p2], 'name'
    yield ok, model['other_name'] is p1
    yield eq, p1.name, 'other_name'
    yield eq, model[p1], 'other_name'

    # Test giving a name to the constructor.
    p3 = GP(EQ(), name='yet_another_name', graph=model)
    yield ok, model['yet_another_name'] is p3
    yield eq, p3.name, 'yet_another_name'
    yield eq, model[p3], 'yet_another_name'


def test_formatting():
    p = 2 * GP(EQ(), 1, graph=Graph())
    yield eq, str(p.display(lambda x: x ** 2)), 'GP(16 * EQ(), 4 * 1)'


def test_sparse_conditioning():
    model = Graph()
    f = GP(EQ().stretch(3), graph=model)
    e = GP(1e-2 * Delta(), graph=model)
    x = np.linspace(0, 5, 10)
    x_new = np.linspace(6, 10, 10)

    y = f(x).sample()

    # Test that noise matrix must indeed be diagonal.
    yield raises, RuntimeError, lambda: SparseObs(f(x), f, f(x), y)

    # Test posterior.
    post_sparse = (f | SparseObs(f(x), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean, \
          'means 1', 1e-6, 1e-6
    yield assert_allclose, post_sparse.var, post_ref.var

    post_sparse = (f | SparseObs(f(x), e, (2 * f + 2)(x), 2 * y + 2))(x_new)
    post_ref = (f | ((2 * f + 2 + e)(x), 2 * y + 2))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean, \
          'means 2', 1e-6, 1e-6
    yield assert_allclose, post_sparse.var, post_ref.var

    post_sparse = (f | SparseObs((2 * f + 2)(x), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean, \
          'means 3', 1e-6, 1e-6
    yield assert_allclose, post_sparse.var, post_ref.var

    # Test ELBO.
    e = GP(1e-2 * Delta(), graph=model)
    yield assert_allclose, \
          SparseObs(f(x), e, f(x), y).elbo, \
          (f + e)(x).logpdf(y)
    yield assert_allclose, \
          SparseObs(f(x), e, (2 * f + 2)(x), 2 * y + 2).elbo, \
          (2 * f + 2 + e)(x).logpdf(2 * y + 2)
    yield assert_allclose, \
          SparseObs((2 * f + 2)(x), e, f(x), y).elbo, \
          (f + e)(x).logpdf(y)

    # Test multiple observations.
    x1 = np.linspace(0, 5, 10)
    x2 = np.linspace(10, 15, 10)
    x_new = np.linspace(6, 9, 10)
    x_ind = np.concatenate((x1, x2, x_new), axis=0)
    y1, y2 = model.sample((f + e)(x1), (f + e)(x2))

    post_sparse = (f | SparseObs(f(x_ind),
                                 (e, f(Unique(x1)), y1),
                                 (e, f(Unique(x2)), y2)))(x_new)
    post_ref = (f | Obs(((f + e)(x1), y1), ((f + e)(x2), y2)))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean
    yield assert_allclose, post_sparse.var, post_ref.var

    # Test multiple observations and multiple inducing points.
    post_sparse = (f | SparseObs((f(x1), f(x2), f(x_new)),
                                 (e, f(Unique(x1)), y1),
                                 (e, f(Unique(x2)), y2)))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean, \
          'means 4', 1e-6, 1e-6
    yield assert_allclose, post_sparse.var, post_ref.var

    # Test multiple inducing points.
    x = np.linspace(0, 5, 10)
    x_new = np.linspace(6, 10, 10)
    x_ind1 = x[:5]
    x_ind2 = x[5:]
    y = model.sample((f + e)(x))

    post_sparse = (f | SparseObs((f(x_ind1), f(x_ind2)), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    yield assert_allclose, post_sparse.mean, post_ref.mean, \
          'means 5', 1e-4, 1e-4
    yield assert_allclose, post_sparse.var, post_ref.var


def test_case_summation_with_itself():
    # Test summing the same GP with itself.
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = p1 + p1 + p1 + p1 + p1
    x = np.linspace(0, 10, 5)[:, None]

    yield assert_allclose, p2(x).var, 25 * p1(x).var
    yield assert_allclose, p2(x).mean, np.zeros((5, 1))

    y = np.random.randn(5, 1)
    yield assert_allclose, p2.condition(p1(x), y)(x).mean, 5 * y


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
    yield assert_allclose, model.kernels[p2, p1](x), np.zeros((n, n))
    yield assert_allclose, model.kernels[p1, p2](x), np.zeros((n, n))

    # Now run through some test cases:

    obs = Obs(p1(x), y1)
    post = (p3 | obs) | ((p2 | obs)(x), y2)
    yield assert_allclose, post(x).mean, y1 + y2

    obs = Obs(p2(x), y2)
    post = (p3 | obs) | ((p1 | obs)(x), y1)
    yield assert_allclose, post(x).mean, y1 + y2

    obs = Obs(p1(x), y1)
    post = (p2 | obs) | ((p3 | obs)(x), y1 + y2)
    yield assert_allclose, post(x).mean, y2

    obs = Obs(p3(x), y1 + y2)
    post = (p2 | obs) | ((p1 | obs)(x), y1)
    yield assert_allclose, post(x).mean, y2

    yield assert_allclose, p3.condition(x, y1 + y2)(x).mean, y1 + y2


def test_case_fd_derivative():
    x = np.linspace(0, 10, 50)[:, None]
    y = np.sin(x)

    model = Graph()
    p = GP(.7 * EQ().stretch(1.), graph=model)
    dp = (p.shift(-1e-3) - p.shift(1e-3)) / 2e-3

    yield le, abs_err(np.cos(x) - dp.condition(p(x), y)(x).mean), 1e-4


def test_case_reflection():
    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = 5 - p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    yield le, abs_err(p2.condition(p(x), y)(x).mean - (5 - y)), 1e-5
    yield le, abs_err(p.condition(p2(x), 5 - y)(x).mean - y), 1e-5

    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = -p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    yield le, abs_err(p2.condition(p(x), y)(x).mean + y), 1e-5
    yield le, abs_err(p.condition(p2(x), -y)(x).mean - y), 1e-5


def test_case_approximate_derivative():
    model = Graph()
    x = np.linspace(0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ().stretch(1.), graph=model)
    dp = p.diff_approx()

    # Test conditioning on function.
    yield le, abs_err(dp.condition(p(x), y)(x).mean - 2), 1e-3

    # Add some regularisation for this test case.
    orig_epsilon = B.epsilon
    B.epsilon = 1e-10

    # Test conditioning on derivative.
    post = p.condition((0, 0), (dp(x), y))
    yield le, abs_err(post(x).mean - x ** 2), 1e-3

    # Set regularisation back.
    B.epsilon = orig_epsilon


def test_case_blr():
    model = Graph()
    x = np.linspace(0, 10, 100)

    slope = GP(1, graph=model)
    intercept = GP(1, graph=model)
    f = slope * (lambda x: x) + intercept
    y = f + 1e-2 * GP(Delta(), graph=model)

    # Sample observations, true slope, and intercept.
    y_obs, true_slope, true_intercept = \
        model.sample(y(x), slope(0), intercept(0))

    # Predict.
    post_slope, post_intercept = (slope, intercept) | Obs(y(x), y_obs)
    mean_slope, mean_intercept = post_slope(0).mean, post_intercept(0).mean

    yield le, np.abs(true_slope[0, 0] - mean_slope[0, 0]), 5e-2
    yield le, np.abs(true_intercept[0, 0] - mean_intercept[0, 0]), 5e-2
