# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno import Means, Kernels, Graph, GP, EQ, model, At, \
    FunctionMean, Linear
# noinspection PyUnresolvedReferences,
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


def test_corner_cases():
    m1 = Graph()
    m2 = Graph()
    p1 = GP(EQ(), graph=m1)
    p2 = GP(EQ(), graph=m2)
    yield raises, RuntimeError, lambda: p1 + p2


def test_means():
    class A(object):
        pass

    vec = Means()

    a1 = A()
    a2 = A()

    vec[a1] = 1
    vec[a2] = 2

    yield ok, vec[a1], 1
    yield ok, vec[a2], 1
    yield ok, vec[id(a1)], 1
    yield ok, vec[id(a2)], 1


def test_kernels():
    class A(object):
        def reverse(self):
            return self

    mat = Kernels()

    # Test diagonal assignments and extractions and conversions.
    a = A()

    mat[a] = 1
    yield eq, mat[a], mat[a, a], 'diagonal'
    yield eq, mat[id(a)], mat[a, a]
    yield eq, mat[a], mat[id(a), a]
    yield eq, mat[id(a)], mat[id(a), a]
    yield eq, mat[a], mat[a, id(a)]
    yield eq, mat[id(a)], mat[a, id(a)]
    yield eq, mat[a], mat[id(a), id(a)]
    yield eq, mat[id(a)], mat[id(a), id(a)]

    mat[a, a] = 2
    yield eq, mat[a], mat[a, a]
    yield eq, mat[id(a)], mat[a, a]
    yield eq, mat[a], mat[id(a), a]
    yield eq, mat[id(a)], mat[id(a), a]
    yield eq, mat[a], mat[a, id(a)]
    yield eq, mat[id(a)], mat[a, id(a)]
    yield eq, mat[a], mat[id(a), id(a)]
    yield eq, mat[id(a)], mat[id(a), id(a)]

    # Check symmetry and reversal of arguments.
    a1 = A()
    a2 = A()

    mat[a1] = (1, 1)
    mat[a1, a2] = (1, 2)
    mat[a2] = (2, 2)

    yield eq, (2, 1), tuple(mat[a2, a1]), 'symmetry'

    mat[a2, a1] = (2, 5)

    yield eq, (2, 5), mat[a2, a1]


def test_sum_other():
    model = Graph()
    p1 = GP(EQ(), FunctionMean(lambda x: x ** 2), graph=model)
    p2 = p1 + 5.
    p3 = 5. + p1

    x = np.random.randn(5, 1)
    yield ok, np.allclose(p1.mean(x) + 5., p2.mean(x))
    yield ok, np.allclose(p1.mean(x) + 5., p3.mean(x))
    yield ok, np.allclose(p1.kernel(x), p2.kernel(x))
    yield ok, np.allclose(p1.kernel(x), p3.kernel(x))
    yield ok, np.allclose(p1.kernel(At(p2)(x), At(p3)(x)),
                          p1.kernel(x))


def test_mul_other():
    model = Graph()
    p1 = GP(EQ(), FunctionMean(lambda x: x ** 2), graph=model)
    p2 = 5. * p1
    p3 = p1 * 5.

    yield raises, NotImplementedError, lambda: p1 * p2

    x = np.random.randn(5, 1)
    yield ok, np.allclose(5. * p1.mean(x), p2.mean(x))
    yield ok, np.allclose(5. * p1.mean(x), p3.mean(x))
    yield ok, np.allclose(25. * p1.kernel(x), p2.kernel(x))
    yield ok, np.allclose(25. * p1.kernel(x), p3.kernel(x))
    yield ok, np.allclose(p1.kernel(At(p2)(x), At(p3)(x)),
                          25. * p1.kernel(x))


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
    eprint(p.length_scale)
    eprint((1 + 2 * 2 ** 2) / (1 + 2 ** 2))

    yield ok, np.allclose(p.length_scale,
                          (1 + 2 * 2 ** 2) / (1 + 2 ** 2)), 'scale'
    yield eq, p.period, 0, 'period'

    yield eq, p3.period, 10, 'period'

    p = p3 + p

    yield eq, p.stationary, True, 'stationary 2'
    yield eq, p.var, 1 + 2 ** 2 + 1, 'var 2'
    yield eq, p.period, 0, 'period 2'

    p = p + GP(Linear(), graph=model)

    yield eq, p.stationary, False, 'stationary 3'


def test_case1():
    # Test summing the same GP with itself.
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = p1 + p1 + p1 + p1 + p1
    x = np.linspace(0, 10, 5)[:, None]

    yield ok, np.allclose(p2(x).var, 25 * p1(x).var), 'variance'
    yield ok, np.allclose(p2(x).mean, np.zeros((5, 1))), 'mean'

    y = np.random.randn(5, 1)
    model.condition(At(p1)(x), y)

    yield ok, np.allclose(p2(x).mean, 5 * y), 'posterior mean'


def test_case2():
    # Test some additive model.

    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ(), graph=model)
    p3 = p1 + p2

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y1 = p1(x).sample()
    y2 = p2(x).sample()

    model.condition(At(p1)(x), y1)
    model.condition(At(p2)(x), y2)
    yield ok, np.allclose(p3(x).mean, y1 + y2)
    model.revert_prior()

    model.condition(At(p2)(x), y2)
    model.condition(At(p1)(x), y1)
    yield ok, np.allclose(p3(x).mean, y1 + y2)
    model.revert_prior()

    model.condition(At(p1)(x), y1)
    model.condition(At(p3)(x), y1 + y2)
    yield ok, np.allclose(p2(x).mean, y2)
    model.revert_prior()

    model.condition(At(p3)(x), y1 + y2)
    model.condition(At(p1)(x), y1)
    yield ok, np.allclose(p2(x).mean, y2)
    model.revert_prior()

    yield ok, np.allclose(p3
                          .condition(At(p1)(x), y1)
                          .condition(At(p2)(x), y2)(x)
                          .mean, y1 + y2)
    p3.revert_prior()

    yield ok, np.allclose(p3
                          .condition(At(p2)(x), y2)
                          .condition(At(p1)(x), y1)(x)
                          .mean, y1 + y2)
    p3.revert_prior()

    yield ok, np.allclose(p2
                          .condition(At(p3)(x), y1 + y2)
                          .condition(At(p1)(x), y1)(x)
                          .mean, y2)
    p3.revert_prior()

    yield ok, np.allclose(p2
                          .condition(At(p1)(x), y1)
                          .condition(At(p3)(x), y1 + y2)(x)
                          .mean, y2)
    p3.revert_prior()

    yield ok, np.allclose(p3.condition(x, y1 + y2)(x).mean, y1 + y2)
