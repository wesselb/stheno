# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B

from stheno.cache import Cache
from stheno.input import Observed
from stheno.kernel import EQ, RQ, Matern12, Matern32, Matern52, Delta, Kernel, \
    Linear, OneKernel, ZeroKernel, PosteriorCrossKernel, PosteriorKernel, \
    ShiftedKernel, TensorProductKernel, VariationalPosteriorCrossKernel
from stheno.random import GPPrimitive
from stheno.matrix import matrix, dense
# noinspection PyUnresolvedReferences
from tests import ok
from . import eq, raises, ok, allclose, assert_allclose


def elwise_generator(k):
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(10, 2)

    # Check `elwise`.
    yield assert_allclose, k.elwise(x1, x2)[:, 0], B.diag(k(x1, x2))
    yield assert_allclose, k.elwise(x1, x2), Kernel.elwise(k, x1, x2)
    yield assert_allclose, k.elwise(x1)[:, 0], B.diag(k(x1))
    yield assert_allclose, k.elwise(x1)[:, 0], B.diag(k(x1))
    yield assert_allclose, k.elwise(x1), Kernel.elwise(k, x1)


def test_corner_cases():
    yield raises, RuntimeError, lambda: Kernel()(1.)


def test_construction():
    k = EQ()

    x = np.random.randn(10, 1)
    c = Cache()

    yield k, x
    yield k, x, c
    yield k, x, x
    yield k, x, x, c

    yield k, Observed(x)
    yield k, Observed(x), c
    yield k, Observed(x), Observed(x)
    yield k, Observed(x), Observed(x), c
    yield k, x, Observed(x)
    yield k, x, Observed(x), c
    yield k, Observed(x), x
    yield k, Observed(x), x, c

    yield k.elwise, x
    yield k.elwise, x, c
    yield k.elwise, x, x
    yield k.elwise, x, x, c

    yield k.elwise, Observed(x)
    yield k.elwise, Observed(x), c
    yield k.elwise, Observed(x), Observed(x)
    yield k.elwise, Observed(x), Observed(x), c
    yield k.elwise, x, Observed(x)
    yield k.elwise, x, Observed(x), c
    yield k.elwise, Observed(x), x
    yield k.elwise, Observed(x), x, c


def test_basic_arithmetic():
    k1 = EQ()
    k2 = RQ(1e-1)
    k3 = Matern12()
    k4 = Matern32()
    k5 = Matern52()
    k6 = Delta()
    k7 = Linear()
    xs1 = np.random.randn(10, 2), np.random.randn(20, 2)
    xs2 = np.random.randn(), np.random.randn()

    yield ok, allclose(k6(xs1[0]), k6(xs1[0], xs1[0])), 'dispatch'
    yield ok, allclose((k1 * k2)(*xs1), k1(*xs1) * k2(*xs1)), 'prod'
    yield ok, allclose((k1 * k2)(*xs2), k1(*xs2) * k2(*xs2)), 'prod 2'
    yield ok, allclose((k3 + k4)(*xs1), k3(*xs1) + k4(*xs1)), 'sum'
    yield ok, allclose((k3 + k4)(*xs2), k3(*xs2) + k4(*xs2)), 'sum 2'
    yield ok, allclose((5. * k5)(*xs1), 5. * k5(*xs1)), 'prod 3'
    yield ok, allclose((5. * k5)(*xs2), 5. * k5(*xs2)), 'prod 4'
    yield ok, allclose((5. + k7)(*xs1), 5. + k7(*xs1)), 'sum 3'
    yield ok, allclose((5. + k7)(*xs2), 5. + k7(*xs2)), 'sum 4'
    yield ok, allclose(k1.stretch(2.)(*xs1),
                       k1(xs1[0] / 2., xs1[1] / 2.)), 'stretch'
    yield ok, allclose(k1.stretch(2.)(*xs2),
                       k1(xs2[0] / 2., xs2[1] / 2.)), 'stretch 2'
    yield ok, allclose(k1.periodic(1.)(*xs1),
                       k1.periodic(1.)(xs1[0], xs1[1] + 5.)), 'periodic'
    yield ok, allclose(k1.periodic(1.)(*xs2),
                       k1.periodic(1.)(xs2[0], xs2[1] + 5.)), 'periodic 2'


def test_reversal():
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)
    x3 = np.random.randn()

    # Test with a stationary and non-stationary kernel.
    for k in [EQ(), Linear()]:
        yield assert_allclose, k(x1), reversed(k)(x1)
        yield assert_allclose, k(x3), reversed(k)(x3)
        yield assert_allclose, k(x1, x2), reversed(k)(x1, x2)
        yield assert_allclose, k(x1, x2), reversed(k)(x2, x1).T

        # Test double reversal does the right thing.
        yield assert_allclose, k(x1), reversed(reversed(k))(x1)
        yield assert_allclose, k(x3), reversed(reversed(k))(x3)
        yield assert_allclose, k(x1, x2), reversed(reversed(k))(x1, x2)
        yield assert_allclose, k(x1, x2), reversed(reversed(k))(x2, x1).T

    # Verify that the kernel has the right properties.
    k = reversed(EQ())
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf

    # Verify that the kernel has the right properties.
    k = reversed(Linear())
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, str(k), 'Reversed(Linear())'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_delta():
    k = Delta()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 0
    yield eq, k.period, np.inf
    yield eq, str(k), 'Delta()'

    yield ok, allclose(k(x1), np.eye(10)), 'same'
    yield ok, allclose(k(x1, x2), np.zeros((10, 5))), 'others'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_eq():
    k = EQ()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'EQ()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_rq():
    k = RQ(1e-1)
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.alpha, 1e-1
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'RQ(0.1)'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_exp():
    k = Matern12()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Exp()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_mat32():
    k = Matern32()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Matern32()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_mat52():
    k = Matern52()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Matern52()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_one():
    k = OneKernel()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T
    yield assert_allclose, k(x1, x2), np.ones((10, 5))

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 0
    yield eq, k.period, 0
    yield eq, str(k), '1'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_zero():
    k = ZeroKernel()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T
    yield assert_allclose, k(x1, x2), np.zeros((10, 5))

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 0
    yield eq, k.length_scale, 0
    yield eq, k.period, 0
    yield eq, str(k), '0'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_linear():
    k = Linear()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, str(k), 'Linear()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_posterior_crosskernel():
    k = PosteriorCrossKernel(
        EQ(), EQ(), EQ(),
        np.random.randn(5, 2), matrix(EQ()(np.random.randn(5, 1)))
    )
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, str(k), 'PosteriorCrossKernel()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    k = PosteriorKernel(GPPrimitive(EQ()), None, None)
    yield eq, str(k), 'PosteriorKernel()'


def test_variational_posterior_crosskernel():
    k = VariationalPosteriorCrossKernel(
        EQ(), EQ(), EQ(),
        np.random.randn(5, 2),
        2 * matrix(EQ()(np.random.randn(5, 1))),
        matrix(EQ()(np.random.randn(5, 1)))
    )
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield assert_allclose, k(x1, x2), k(x2, x1).T

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, str(k), 'VariationalPosteriorCrossKernel()'

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    yield eq, k.stationary, True
    yield assert_allclose, k.length_scale, (1 * 2 + 3 * 5) / 4
    yield eq, k.period, np.inf
    yield eq, k.var, 4

    yield assert_allclose, (EQ() + EQ()).length_scale, 1
    yield assert_allclose, (EQ().stretch(2) + EQ().stretch(2)).length_scale, 2

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_stretch():
    k = EQ().stretch(2)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 2
    yield eq, k.period, np.inf
    yield eq, k.var, 1

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    k = EQ().stretch(1, 2)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 1

    # Check passing in a list.
    k = EQ().stretch([1, 2])
    yield k, np.random.randn(10, 2)


def test_periodic():
    k = EQ().stretch(2).periodic(3)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 2
    yield eq, k.period, 3
    yield eq, k.var, 1

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    k = 5 * k.stretch(5)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 10
    yield eq, k.period, 15
    yield eq, k.var, 5

    # Check passing in a list.
    k = EQ().periodic([1, 2])
    yield k, np.random.randn(10, 2)


def test_scaled():
    k = 2 * EQ()

    yield eq, k.stationary, True
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    # test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_shifted():
    k = ShiftedKernel(2 * EQ(), 5)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    # test `elwise`.
    for x in elwise_generator(k):
        yield x

    k = (2 * EQ()).shift(5, 6)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    # Check computation.
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)
    k = Linear()
    yield assert_allclose, k.shift(5)(x1, x2), k(x1 - 5, x2 - 5)

    # Check passing in a list.
    k = Linear().shift([1, 2])
    yield k, np.random.randn(10, 2)


def test_product():
    k = (2 * EQ().stretch(10)) * (3 * RQ(1e-2).stretch(20))

    yield eq, k.stationary, True
    yield eq, k.length_scale, 10
    yield eq, k.period, np.inf
    yield eq, k.var, 6

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x


def test_selected():
    k = (2 * EQ().stretch(5)).select(0)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 5
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    k = (2 * EQ().stretch(5)).select(2, 3)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 5
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    k = (2 * EQ().stretch([1, 2, 3])).select(0, 2)

    yield eq, k.stationary, True
    yield assert_allclose, k.length_scale, [1, 3]
    yield assert_allclose, k.period, [np.inf, np.inf]
    yield eq, k.var, 2

    k = (2 * EQ().periodic([1, 2, 3])).select(1, 2)

    yield eq, k.stationary, True
    yield assert_allclose, k.length_scale, [1, 1]
    yield assert_allclose, k.period, [2, 3]
    yield eq, k.var, 2

    k = (2 * EQ().stretch([1, 2, 3])).select((0, 2), (1, 2))

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 2

    k = (2 * EQ().periodic([1, 2, 3])).select((0, 2), (1, 2))

    yield eq, k.stationary, False
    yield eq, k.length_scale, 1
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 2

    # Test that computation is valid.
    k1 = EQ().select(1, 2)
    k2 = EQ()
    x = np.random.randn(10, 3)
    yield assert_allclose, k1(x), k2(x[:, [1, 2]])


def test_input_transform():
    k = Linear().transform(lambda x, c: x - 5)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    # Test computation of the kernel.
    k = Linear()
    x1, x2 = np.random.randn(10, 2), np.random.randn(10, 2)

    k2 = k.transform(lambda x, c: x ** 2)
    k3 = k.transform(lambda x, c: x ** 2, lambda x, c: x - 5)

    yield assert_allclose, k(x1 ** 2, x2 ** 2), k2(x1, x2)
    yield assert_allclose, k(x1 ** 2, x2 - 5), k3(x1, x2)


def test_tensor_product():
    k = TensorProductKernel(lambda x: B.sum(x ** 2, axis=1)[:, None])

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period

    # Test `elwise`.
    for x in elwise_generator(k):
        yield x

    # Check computation of kernel.
    k = TensorProductKernel(lambda x: x)
    x1 = np.linspace(0, 1, 100)[:, None]
    x2 = np.linspace(0, 1, 50)[:, None]

    yield assert_allclose, k(x1), x1 * x1.T
    yield assert_allclose, k(x1, x2), x1 * x2.T

    k = TensorProductKernel(lambda x: x ** 2)

    yield assert_allclose, k(x1), x1 ** 2 * (x1 ** 2).T
    yield assert_allclose, k(x1, x2), (x1 ** 2) * (x2 ** 2).T


def test_derivative():
    # First, check properties.
    k = EQ().diff(0)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period

    yield raises, RuntimeError, lambda: EQ().diff(None, None)(1)

    # Third, check computation.
    B.backend_to_tf()
    s = B.Session()

    # Test derivative of kernel EQ.
    k = EQ()
    x1 = B.array(np.random.randn(10, 1))
    x2 = B.array(np.random.randn(5, 1))

    # Test derivative with respect to first input.
    ref = s.run(-dense(k(x1, x2)) * (x1 - B.transpose(x2)))
    yield assert_allclose, s.run(dense(k.diff(0, None)(x1, x2))), ref
    ref = s.run(-dense(k(x1)) * (x1 - B.transpose(x1)))
    yield assert_allclose, s.run(dense(k.diff(0, None)(x1))), ref

    # Test derivative with respect to second input.
    ref = s.run(-dense(k(x1, x2)) * (B.transpose(x2) - x1))
    yield assert_allclose, s.run(dense(k.diff(None, 0)(x1, x2))), ref
    ref = s.run(-dense(k(x1)) * (B.transpose(x1) - x1))
    yield assert_allclose, s.run(dense(k.diff(None, 0)(x1))), ref

    # Test derivative with respect to both inputs.
    ref = s.run(dense(k(x1, x2)) * (1 - (x1 - B.transpose(x2)) ** 2))
    yield assert_allclose, s.run(dense(k.diff(0, 0)(x1, x2))), ref
    yield assert_allclose, s.run(dense(k.diff(0)(x1, x2))), ref
    ref = s.run(dense(k(x1)) * (1 - (x1 - B.transpose(x1)) ** 2))
    yield assert_allclose, s.run(dense(k.diff(0, 0)(x1))), ref
    yield assert_allclose, s.run(dense(k.diff(0)(x1))), ref

    # Test derivative of kernel Linear.
    k = Linear()
    x1 = B.array(np.random.randn(10, 1))
    x2 = B.array(np.random.randn(5, 1))

    # Test derivative with respect to first input.
    ref = s.run(B.ones((10, 5), dtype=np.float64) * B.transpose(x2))
    yield assert_allclose, s.run(dense(k.diff(0, None)(x1, x2))), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64) * B.transpose(x1))
    yield assert_allclose, s.run(dense(k.diff(0, None)(x1))), ref

    # Test derivative with respect to second input.
    ref = s.run(B.ones((10, 5), dtype=np.float64) * x1)
    yield assert_allclose, s.run(dense(k.diff(None, 0)(x1, x2))), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64) * x1)
    yield assert_allclose, s.run(dense(k.diff(None, 0)(x1))), ref

    # Test derivative with respect to both inputs.
    ref = s.run(B.ones((10, 5), dtype=np.float64))
    yield assert_allclose, s.run(dense(k.diff(0, 0)(x1, x2))), ref
    yield assert_allclose, s.run(dense(k.diff(0)(x1, x2))), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64))
    yield assert_allclose, s.run(dense(k.diff(0, 0)(x1))), ref
    yield assert_allclose, s.run(dense(k.diff(0)(x1))), ref

    s.close()
    B.backend_to_np()
