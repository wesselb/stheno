# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import tensorflow as tf
from lab import B
from plum import type_parameter

from stheno.graph import Graph, GP, Obs, SparseObs
from stheno.input import At, Unique
from stheno.kernel import (
    Linear,
    EQ,
    Delta,
    Exp,
    RQ,
    ZeroKernel,
    OneKernel,
    ScaledKernel
)
from stheno.mean import TensorProductMean, ZeroMean, ScaledMean, OneMean
from stheno.random import Normal
from .util import allclose


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
    with pytest.raises(RuntimeError):
        p1 + p2
    with pytest.raises(NotImplementedError):
        p1 + p1(x)
    with pytest.raises(NotImplementedError):
        p1 * p1(x)
    assert str(GP(EQ(), graph=m1)) == 'GP(EQ(), 0)'


def test_construction():
    model = Graph()
    p = GP(EQ(), graph=model)

    x = np.random.randn(10, 1)

    p.mean(x)
    p.mean(p(x))

    p.kernel(x)
    p.kernel(p(x))
    p.kernel(x, x)
    p.kernel(p(x), x)
    p.kernel(x, p(x))
    p.kernel(p(x), p(x))

    p.kernel.elwise(x)
    p.kernel.elwise(p(x))
    p.kernel.elwise(x, x)
    p.kernel.elwise(p(x), x)
    p.kernel.elwise(x, p(x))
    p.kernel.elwise(p(x), p(x))

    # Test resolution of kernel and mean.
    k = EQ()
    m = TensorProductMean(lambda x: x ** 2)

    assert isinstance(GP(k, graph=model).mean, ZeroMean)
    assert isinstance(GP(k, 5, graph=model).mean, ScaledMean)
    assert isinstance(GP(k, 1, graph=model).mean, OneMean)
    assert isinstance(GP(k, 0, graph=model).mean, ZeroMean)
    assert isinstance(GP(k, m, graph=model).mean, TensorProductMean)
    assert isinstance(GP(k, graph=model).kernel, EQ)
    assert isinstance(GP(5, graph=model).kernel, ScaledKernel)
    assert isinstance(GP(1, graph=model).kernel, OneKernel)
    assert isinstance(GP(0, graph=model).kernel, ZeroKernel)

    # Test construction of finite-dimensional distribution.
    d = GP(k, m, graph=model)(x)
    allclose(d.var, k(x))
    allclose(d.mean, m(x))


def test_sum_other():
    model = Graph()
    p1 = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    p2 = p1 + 5.
    p3 = 5. + p1
    p4 = model.sum(5., p1)

    x = np.random.randn(5, 1)
    allclose(p1.mean(x) + 5., p2.mean(x))
    allclose(p1.mean(x) + 5., p3.mean(x))
    allclose(p1.mean(x) + 5., p4.mean(x))
    allclose(p1.kernel(x), p2.kernel(x))
    allclose(p1.kernel(x), p3.kernel(x))
    allclose(p1.kernel(x), p4.kernel(x))
    allclose(p1.kernel(p2(x), p3(x)), p1.kernel(x))
    allclose(p1.kernel(p2(x), p4(x)), p1.kernel(x))

    # Check that a `GP` cannot be summed with a `Normal`.
    with pytest.raises(NotImplementedError):
        p1 + Normal(np.eye(3))
    with pytest.raises(NotImplementedError):
        Normal(np.eye(3)) + p1


def test_mul_other():
    model = Graph()
    p1 = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    p2 = 5. * p1
    p3 = p1 * 5.

    x = np.random.randn(5, 1)
    allclose(5. * p1.mean(x), p2.mean(x))
    allclose(5. * p1.mean(x), p3.mean(x))
    allclose(25. * p1.kernel(x), p2.kernel(x))
    allclose(25. * p1.kernel(x), p3.kernel(x))
    allclose(model.kernels[p2, p3](x, x), 25. * p1.kernel(x))

    # Check that a `GP` cannot be multiplied with a `Normal`.
    with pytest.raises(NotImplementedError):
        p1 * Normal(np.eye(3))
    with pytest.raises(NotImplementedError):
        Normal(np.eye(3)) * p1


def test_shorthands():
    model = Graph()
    p = GP(EQ(), graph=model)

    # Construct a normal distribution that serves as in input.
    x = p(1)
    assert isinstance(x, At)
    assert type_parameter(x) is p
    assert x.get() == 1
    assert str(p(x)) == '{}({})'.format(str(p), str(x))
    assert repr(p(x)) == '{}({})'.format(repr(p), repr(x))

    # Construct a normal distribution that does not serve as an input.
    x = Normal(np.ones((1, 1)))
    with pytest.raises(RuntimeError):
        type_parameter(x)
    with pytest.raises(RuntimeError):
        x.get()
    with pytest.raises(RuntimeError):
        p | (x, 1)

    # Test shorthands for stretching and selection.
    p = GP(EQ(), graph=Graph())
    assert str(p > 2) == str(p.stretch(2))
    assert str(p[0]) == str(p.select(0))


def test_properties():
    model = Graph()

    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ().stretch(2), graph=model)
    p3 = GP(EQ().periodic(10), graph=model)

    p = p1 + 2 * p2

    assert p.stationary == True, 'stationary'
    assert p.var == 1 + 2 ** 2, 'var'
    allclose(p.length_scale, (1 + 2 * 2 ** 2) / (1 + 2 ** 2))
    assert p.period == np.inf, 'period'

    assert p3.period == 10, 'period'

    p = p3 + p

    assert p.stationary == True, 'stationary 2'
    assert p.var == 1 + 2 ** 2 + 1, 'var 2'
    assert p.period == np.inf, 'period 2'

    p = p + GP(Linear(), graph=model)

    assert p.stationary == False, 'stationary 3'


def test_marginals():
    model = Graph()
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    x = np.linspace(0, 5, 10)

    # Check that `marginals` outputs the right thing.
    mean, lower, upper = p(x).marginals()
    var = B.diag(p.kernel(x))
    allclose(mean, p.mean(x)[:, 0])
    allclose(lower, p.mean(x)[:, 0] - 2 * var ** .5)
    allclose(upper, p.mean(x)[:, 0] + 2 * var ** .5)

    # Test correctness.
    y = p(x).sample()
    mean, lower, upper = (p | (x, y))(x).marginals()
    allclose(mean, y[:, 0])
    assert B.mean(B.abs(upper - lower)) <= 1e-5

    mean, lower, upper = (p | (x, y))(x + 100).marginals()
    allclose(mean, p.mean(x + 100)[:, 0])
    allclose(upper - lower, 4 * np.ones(10))


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
    allclose(obs1.y, obs2.y)
    allclose(obs1.K_x, obs2.K_x)

    obs3 = Obs((p(x), y), (p1(x), y1))
    obs4 = Obs((x, y), (p1(x), y1), ref=p)
    allclose(obs3.y, obs4.y)
    allclose(obs3.K_x, obs4.K_x)

    def assert_equal_mean_var(x, *ys):
        for y in ys:
            allclose(x.mean, y.mean)
            allclose(x.var, y.var)

    assert_equal_mean_var(p.condition(x, y)(x),
                          p.condition(p(x), y)(x),
                          (p | (x, y))(x),
                          (p | (p(x), y))(x),
                          p.condition(obs1)(x),
                          p.condition(obs2)(x),
                          (p | obs1)(x),
                          (p | obs2)(x))

    assert_equal_mean_var(p.condition((x, y), (p1(x), y1))(x),
                          p.condition((p(x), y), (p1(x), y1))(x),
                          (p | [(x, y), (p1(x), y1)])(x),
                          (p | [(p(x), y), (p1(x), y1)])(x),
                          p.condition(obs3)(x),
                          p.condition(obs4)(x),
                          (p | obs3)(x),
                          (p | obs4)(x))

    # Check conditioning multiple processes at once.
    p1_post, p2_post, p_post = (p1, p2, p) | obs1
    p1_post, p2_post, p_post = p1_post(x), p2_post(x), p_post(x)
    p1_post2, p2_post2, p_post2 = (p1 | obs1)(x), (p2 | obs1)(x), (p | obs1)(x)

    allclose(p1_post.mean, p1_post2.mean)
    allclose(p1_post.var, p1_post2.var)
    allclose(p2_post.mean, p2_post2.mean)
    allclose(p2_post.var, p2_post2.var)
    allclose(p_post.mean, p_post2.mean)
    allclose(p_post.var, p_post2.var)

    # Test `At` check.
    with pytest.raises(ValueError):
        Obs(0, 0)
    with pytest.raises(ValueError):
        Obs((0, 0), (0, 0))
    with pytest.raises(ValueError):
        SparseObs(0, p, (0, 0))

    # Test that `Graph.logpdf` takes an `Observations` object.
    obs = Obs(p(x), y)
    assert model.logpdf(obs) == p(x).logpdf(y)


def test_stretching():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    assert str(p.stretch(1)) == 'GP(EQ() > 1, <lambda> > 1)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.stretch(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    allclose(post(x / 5).mean, y)
    assert abs_err(B.diag(post(x / 5).var)) <= 1e-10

    post = p2.condition(p(x), y)
    allclose(post(x * 5).mean, y)
    assert abs_err(B.diag(post(x * 5).var)) <= 1e-10


def test_shifting():
    model = Graph()

    # Test construction:
    p = GP(Linear(), TensorProductMean(lambda x: x ** 2), graph=model)
    assert str(p.shift(1)) == 'GP(Linear() shift 1, <lambda> shift 1)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.shift(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    allclose(post(x - 5).mean, y)
    assert abs_err(B.diag(post(x - 5).var)) <= 1e-10

    post = p2.condition(p(x), y)
    allclose(post(x + 5).mean, y)
    assert abs_err(B.diag(post(x + 5).var)) <= 1e-10


def test_input_transform():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    assert str(p.transform(lambda x: x)) == \
           'GP(EQ() transform <lambda>, <lambda> transform <lambda>)'

    # Test case:
    p = GP(EQ(), graph=model)
    p2 = p.transform(lambda x: x / 5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    post = p.condition(p2(x), y)
    allclose(post(x / 5).mean, y)
    assert abs_err(B.diag(post(x / 5).var)) <= 1e-10

    post = p2.condition(p(x), y)
    allclose(post(x * 5).mean, y)
    assert abs_err(B.diag(post(x * 5).var)) <= 1e-10


def test_selection():
    model = Graph()

    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=model)
    assert str(p.select(1)) == 'GP(EQ() : [1], <lambda> : [1])'
    assert str(p.select(1, 2)) == 'GP(EQ() : [1, 2], <lambda> : [1, 2])'

    # Test case:
    p = GP(EQ(), graph=model)  # 1D
    p2 = p.select(0)  # 2D

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    x1 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    x2 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    y = p2(x).sample()

    post = p.condition(p2(x1), y)
    allclose(post(x).mean, y)
    assert abs_err(B.diag(post(x).var)) <= 1e-10

    post = p.condition(p2(x2), y)
    allclose(post(x).mean, y)
    assert abs_err(B.diag(post(x).var)) <= 1e-10

    post = p2.condition(p(x), y)
    allclose(post(x1).mean, y)
    allclose(post(x2).mean, y)
    assert abs_err(B.diag(post(x1).var)) <= 1e-10
    assert abs_err(B.diag(post(x2).var)) <= 1e-10


def test_derivative():
    # Test construction:
    p = GP(EQ(), TensorProductMean(lambda x: x ** 2), graph=Graph())
    assert str(p.diff(1)) == 'GP(d(1) EQ(), d(1) <lambda>)'

    # Test case:
    model = Graph()
    x = B.linspace(tf.float64, 0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ(), graph=model)
    dp = p.diff()

    # Test conditioning on function.
    assert abs_err(dp.condition(p(x), y)(x).mean, 2) <= 1e-3

    # Test conditioning on derivative.
    post = p.condition((B.cast(tf.float64, 0),
                        B.cast(tf.float64, 0)), (dp(x), y))
    assert abs_err(post(x).mean, x ** 2) <= 1e-3


def test_multi_sample():
    model = Graph()
    p1 = GP(0, 1, graph=model)
    p2 = GP(0, 2, graph=model)
    p3 = GP(0, 3, graph=model)

    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(0, 1, 20)
    x3 = np.linspace(0, 1, 30)

    s1, s2, s3 = model.sample(p1(x1), p2(x2), p3(x3))

    assert s1.shape == (10, 1)
    assert s2.shape == (20, 1)
    assert s3.shape == (30, 1)
    assert np.shape(model.sample(p1(x1))) == (10, 1)

    assert abs_err(s1 - 1) <= 1e-4
    assert abs_err(s2 - 2) <= 1e-4
    assert abs_err(s3 - 3) <= 1e-4


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

    allclose(post1.mean, post2.mean, desc='means 1', atol=1e-6, rtol=1e-6)
    allclose(post1.mean, post3.mean, desc='means 2', atol=1e-6, rtol=1e-6)
    allclose(post1.var, post2.var)
    allclose(post1.var, post3.var)


def test_multi_logpdf():
    model = Graph()

    p1 = GP(EQ(), graph=model)
    p2 = GP(2 * Exp(), graph=model)
    p3 = p1 + p2

    x1 = np.linspace(0, 2, 5)
    x2 = np.linspace(1, 3, 6)
    x3 = np.linspace(2, 4, 7)

    y1, y2, y3 = model.sample(p1(x1), p2(x2), p3(x3))

    # Test case that only one process is fed.
    allclose(p1(x1).logpdf(y1), model.logpdf(p1(x1), y1))
    allclose(p1(x1).logpdf(y1), model.logpdf((p1(x1), y1)))

    # Test that all inputs must be specified.
    with pytest.raises(ValueError):
        model.logpdf((x1, y1), (p2(x2), y2))
    with pytest.raises(ValueError):
        model.logpdf((p1(x1), y1), (x2, y2))

    # Compute the logpdf with the product rule.
    logpdf1 = p1(x1).logpdf(y1) + p2(x2).logpdf(y2) + \
              (p3 | ((p1(x1), y1), (p2(x2), y2)))(x3).logpdf(y3)
    logpdf2 = model.logpdf((p1(x1), y1), (p2(x2), y2), (p3(x3), y3))
    allclose(logpdf1, logpdf2)


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
    assert rel_err(post(x).mean, s1 * s2) <= 1e-2

    # Perform division.
    cur_epsilon = B.epsilon
    B.epsilon = 1e-8
    post = p2.condition((p1(x), s1), (p_prod(x), s1 * s2))
    assert rel_err(post(x).mean, s2) <= 1e-2
    B.epsilon = cur_epsilon

    # Check graph check.
    model2 = Graph()
    p3 = GP(EQ(), graph=model2)
    with pytest.raises(RuntimeError):
        p3 * p1


def test_naming():
    model = Graph()

    p1 = GP(EQ(), 1, graph=model)
    p2 = GP(EQ(), 2, graph=model)

    # Test setting and getting names.
    p1.name = 'name'

    assert model['name'] is p1
    assert p1.name == 'name'
    assert model[p1] == 'name'
    with pytest.raises(KeyError):
        model['other_name']
    with pytest.raises(KeyError):
        model[p2]

    # Check that names can not be doubly assigned.
    def doubly_assign():
        p2.name = 'name'

    with pytest.raises(RuntimeError):
        doubly_assign()

    # Move name to other GP.
    p1.name = 'other_name'
    p2.name = 'name'

    # Check that everything has been properly assigned.
    assert model['name'] is p2
    assert p2.name == 'name'
    assert model[p2] == 'name'
    assert model['other_name'] is p1
    assert p1.name == 'other_name'
    assert model[p1] == 'other_name'

    # Test giving a name to the constructor.
    p3 = GP(EQ(), name='yet_another_name', graph=model)
    assert model['yet_another_name'] is p3
    assert p3.name == 'yet_another_name'
    assert model[p3] == 'yet_another_name'


def test_formatting():
    p = 2 * GP(EQ(), 1, graph=Graph())
    assert str(p.display(lambda x: x ** 2)) == 'GP(16 * EQ(), 4 * 1)'


def test_sparse_conditioning():
    model = Graph()
    f = GP(EQ().stretch(3), graph=model)
    e = GP(1e-2 * Delta(), graph=model)
    x = np.linspace(0, 5, 10)
    x_new = np.linspace(6, 10, 10)

    y = f(x).sample()

    # Test that noise matrix must indeed be diagonal.
    with pytest.raises(RuntimeError):
        SparseObs(f(x), f, f(x), y).elbo

    # Test posterior.
    post_sparse = (f | SparseObs(f(x), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    allclose(post_sparse.mean, post_ref.mean, desc='means 1', atol=1e-6,
             rtol=1e-6)
    allclose(post_sparse.var, post_ref.var)

    post_sparse = (f | SparseObs(f(x), e, (2 * f + 2)(x), 2 * y + 2))(x_new)
    post_ref = (f | ((2 * f + 2 + e)(x), 2 * y + 2))(x_new)
    allclose(post_sparse.mean, post_ref.mean, desc='means 2', atol=1e-6,
             rtol=1e-6)
    allclose(post_sparse.var, post_ref.var)

    post_sparse = (f | SparseObs((2 * f + 2)(x), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    allclose(post_sparse.mean, post_ref.mean, desc='means 3', atol=1e-6,
             rtol=1e-6)
    allclose(post_sparse.var, post_ref.var)

    # Test ELBO.
    e = GP(1e-2 * Delta(), graph=model)
    allclose(SparseObs(f(x), e, f(x), y).elbo, (f + e)(x).logpdf(y))
    allclose(SparseObs(f(x), e, (2 * f + 2)(x), 2 * y + 2).elbo,
             (2 * f + 2 + e)(x).logpdf(2 * y + 2))
    allclose(SparseObs((2 * f + 2)(x), e, f(x), y).elbo, (f + e)(x).logpdf(y))

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
    allclose(post_sparse.mean, post_ref.mean)
    allclose(post_sparse.var, post_ref.var)

    # Test multiple observations and multiple inducing points.
    post_sparse = (f | SparseObs((f(x1), f(x2), f(x_new)),
                                 (e, f(Unique(x1)), y1),
                                 (e, f(Unique(x2)), y2)))(x_new)
    allclose(post_sparse.mean, post_ref.mean, desc='means 4', atol=1e-6,
             rtol=1e-6)
    allclose(post_sparse.var, post_ref.var)

    # Test multiple inducing points.
    x = np.linspace(0, 5, 10)
    x_new = np.linspace(6, 10, 10)
    x_ind1 = x[:5]
    x_ind2 = x[5:]
    y = model.sample((f + e)(x))

    post_sparse = (f | SparseObs((f(x_ind1), f(x_ind2)), e, f(x), y))(x_new)
    post_ref = (f | ((f + e)(x), y))(x_new)
    allclose(post_sparse.mean, post_ref.mean, desc='means 5', atol=1e-4,
             rtol=1e-4)
    allclose(post_sparse.var, post_ref.var)

    # Test caching of mean.
    obs = SparseObs(f(x), e, f(x), y)
    mu = obs.mu
    allclose(mu, obs.mu)

    # Test caching of corrective kernel parameter.
    obs = SparseObs(f(x), e, f(x), y)
    A = obs.A
    allclose(A, obs.A)

    # Test caching of elbo.
    obs = SparseObs(f(x), e, f(x), y)
    elbo = obs.elbo
    allclose(elbo, obs.elbo)

    # Test that `Graph.logpdf` takes an `SparseObservations` object.
    obs = SparseObs(f(x), e, f(x), y)
    allclose(model.logpdf(obs), (f + e)(x).logpdf(y))


def test_case_summation_with_itself():
    # Test summing the same GP with itself.
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = p1 + p1 + p1 + p1 + p1
    x = np.linspace(0, 10, 5)[:, None]

    allclose(p2(x).var, 25 * p1(x).var)
    allclose(p2(x).mean, np.zeros((5, 1)))

    y = np.random.randn(5, 1)
    allclose(p2.condition(p1(x), y)(x).mean, 5 * y)


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
    allclose(model.kernels[p2, p1](x), np.zeros((n, n)))
    allclose(model.kernels[p1, p2](x), np.zeros((n, n)))

    # Now run through some test cases:

    obs = Obs(p1(x), y1)
    post = (p3 | obs) | ((p2 | obs)(x), y2)
    allclose(post(x).mean, y1 + y2)

    obs = Obs(p2(x), y2)
    post = (p3 | obs) | ((p1 | obs)(x), y1)
    allclose(post(x).mean, y1 + y2)

    obs = Obs(p1(x), y1)
    post = (p2 | obs) | ((p3 | obs)(x), y1 + y2)
    allclose(post(x).mean, y2)

    obs = Obs(p3(x), y1 + y2)
    post = (p2 | obs) | ((p1 | obs)(x), y1)
    allclose(post(x).mean, y2)

    allclose(p3.condition(x, y1 + y2)(x).mean, y1 + y2)


def test_case_fd_derivative():
    x = np.linspace(0, 10, 50)[:, None]
    y = np.sin(x)

    model = Graph()
    p = GP(.7 * EQ().stretch(1.), graph=model)
    dp = (p.shift(-1e-3) - p.shift(1e-3)) / 2e-3

    assert abs_err(np.cos(x) - dp.condition(p(x), y)(x).mean) <= 1e-4


def test_case_reflection():
    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = 5 - p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    assert abs_err(p2.condition(p(x), y)(x).mean - (5 - y)) <= 1e-5
    assert abs_err(p.condition(p2(x), 5 - y)(x).mean - y) <= 1e-5

    model = Graph()
    p = GP(EQ(), graph=model)
    p2 = -p

    x = np.linspace(0, 1, 10)[:, None]
    y = p(x).sample()

    assert abs_err(p2.condition(p(x), y)(x).mean + y) <= 1e-5
    assert abs_err(p.condition(p2(x), -y)(x).mean - y) <= 1e-5


def test_case_approximate_derivative():
    model = Graph()
    x = np.linspace(0, 1, 100)[:, None]
    y = 2 * x

    p = GP(EQ().stretch(1.), graph=model)
    dp = p.diff_approx()

    # Test conditioning on function.
    assert abs_err(dp.condition(p(x), y)(x).mean - 2) <= 1e-3

    # Add some regularisation for this test case.
    orig_epsilon = B.epsilon
    B.epsilon = 1e-10

    # Test conditioning on derivative.
    post = p.condition((0, 0), (dp(x), y))
    assert abs_err(post(x).mean - x ** 2) <= 1e-3

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

    assert np.abs(true_slope[0, 0] - mean_slope[0, 0]) <= 5e-2
    assert np.abs(true_intercept[0, 0] - mean_intercept[0, 0]) <= 5e-2
