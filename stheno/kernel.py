# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import operator
from numbers import Number

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

from .cache import cache, Cache
from .field import add, mul, dispatch, Type, PrimitiveType, \
    ZeroType, OneType, ScaledType, ProductType, SumType, StretchedType, \
    WrappedType, ShiftedType, SelectedType, InputTransformedType, JoinType, \
    DerivativeType, broadcast
from .input import Input

__all__ = ['Kernel', 'OneKernel', 'ZeroKernel', 'ScaledKernel', 'EQ', 'RQ',
           'Matern12', 'Exp', 'Matern32', 'Matern52', 'Delta', 'Linear',
           'DerivativeKernel']

log = logging.getLogger(__name__)


def expand(xs):
    return xs * 2 if len(xs) == 1 else xs


class Kernel(Type, Referentiable):
    """Kernel function.

    Kernels can be added and multiplied.
    """
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    def __call__(self, x, y, cache):
        """Construct the kernel for design matrices of points.

        Args:
            x (design matrix): First argument.
            y (design matrix, optional): Second argument. Defaults to first
                argument.
            cache (instance of :class:`.cache.Cache`, optional): Cache.

        Returns:
            Kernel matrix.
        """
        raise NotImplementedError('Could not resolve kernel arguments.')

    @dispatch(object)
    def __call__(self, x):
        return self(x, x, Cache())

    @dispatch.multi((object, Cache), (Input, Cache))
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @dispatch(Input, Input)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @dispatch(Input, Input, Cache)
    def __call__(self, x, y, cache):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get(), y.get(), cache)

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor, optional): Period. Defaults to `1`.
        """
        return periodicise(self, period)

    def __reversed__(self):
        """Reverse the arguments of the kernel."""
        return reverse(self)

    @property
    def stationary(self):
        """Stationarity of the kernel"""
        try:
            return self._stationary_cache
        except AttributeError:
            self._stationary_cache = self._stationary
            return self._stationary_cache

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        """Variance of the kernel. Returns `np.nan` if the variance is
        undefined or cannot be determined.
        """
        return 1

    @property
    def length_scale(self):
        """Approximation to the length scale of the kernel. Returns `np.nan`
        if the length scale is undefined or cannot be determined.
        """
        return 1

    @property
    def period(self):
        """Period of the kernel. Returns `np.nan` is the period is undefined
        or cannot be determined.
        """
        return 0


class OneKernel(Kernel, OneType, Referentiable):
    """Constant kernel of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.ones((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def length_scale(self):
        return np.inf


class ZeroKernel(Kernel, ZeroType, Referentiable):
    """Constant kernel of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.zeros((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def var(self):
        return 0

    @property
    def length_scale(self):
        return 0


class ScaledKernel(Kernel, ScaledType, Referentiable):
    """Scaled kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.multiply(B.cast(self.scale, dtype=B.dtype(x)),
                          self[0](x, y, B))

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self.scale * self[0].var

    @property
    def length_scale(self):
        return self[0].length_scale

    @property
    def period(self):
        return self[0].period


class SumKernel(Kernel, SumType, Referentiable):
    """Sum of kernels."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.add(self[0](x, y, B), self[1](x, y, B))

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary

    @property
    def var(self):
        return self[0].var + self[1].var

    @property
    def length_scale(self):
        return (self[0].var * self[0].length_scale +
                self[1].var * self[1].length_scale) / self.var


class ProductKernel(Kernel, ProductType, Referentiable):
    """Product of two kernels."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.multiply(self[0](x, y, B), self[1](x, y, B))

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary

    @property
    def var(self):
        return self[0].var * self[1].var

    @property
    def length_scale(self):
        return B.minimum(self[0].length_scale, self[1].length_scale)


class StretchedKernel(Kernel, StretchedType, Referentiable):
    """Stretched kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        stretches = expand(self.stretches)
        return self[0](B.divide(x, stretches[0]), B.divide(y, stretches[1]), B)

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        if len(self.stretches) == 1:
            return self[0].length_scale * self.stretches[0]
        else:
            return np.nan

    @property
    def period(self):
        if len(self.stretches) == 1:
            return self[0].period * self.stretches[0]
        else:
            return np.nan


class ShiftedKernel(Kernel, ShiftedType, Referentiable):
    """Shifted kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        shifts = expand(self.shifts)
        return self[0](B.subtract(x, shifts[0]), B.subtract(y, shifts[1]), B)

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        return self[0].length_scale

    @property
    def period(self):
        return self[0].period


class SelectedKernel(Kernel, SelectedType, Referentiable):
    """Kernel with particular input dimensions selected."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        dims = expand(self.dims)
        return self[0](B.take(x, dims[0], axis=1),
                       B.take(y, dims[1], axis=1), B)

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        length_scale = self[0].length_scale
        if B.is_scalar(length_scale):
            return length_scale
        else:
            if len(self.dims) == 1:
                return B.take(length_scale, self.dims[0])
            else:
                return np.nan

    @property
    def period(self):
        period = self[0].period
        if B.is_scalar(period):
            return period
        else:
            if len(self.dims) == 1:
                return B.take(period, self.dims[0])
            else:
                return np.nan


class InputTransformedKernel(Kernel, InputTransformedType, Referentiable):
    """Input-transformed kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        fs = expand(self.fs)
        return self[0](fs[0](x, B), fs[1](y, B), B)

    @property
    def _stationary(self):
        return False

    @property
    def var(self):
        return np.nan

    @property
    def length_scale(self):
        return np.nan


class PeriodicKernel(Kernel, WrappedType, Referentiable):
    """Periodic kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, period):
        WrappedType.__init__(self, k)
        self._period = period

    @dispatch(Number, Number, Cache)
    @cache
    def __call__(self, x, y, B):
        def feat_map(z):
            z = B.divide(B.multiply(B.multiply(z, 2), B.pi), self.period)
            return B.array([[B.sin(z), B.cos(z)]])

        return self[0](feat_map(x), feat_map(y), B)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        def feat_map(z):
            z = B.divide(B.multiply(B.multiply(z, 2), B.pi), self.period)
            return B.concat((B.sin(z), B.cos(z)), axis=1)

        return self[0](feat_map(x), feat_map(y), B)

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        return self[0].length_scale

    @property
    def period(self):
        return self._period

    def display(self, t):
        return '{} per {}'.format(t, self._period)


class EQ(Kernel, PrimitiveType, Referentiable):
    """Exponentiated quadratic kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.exp(B.multiply(B.cast(-.5, dtype=B.dtype(x)),
                                B.pw_dists2(x, y)))

    def __str__(self):
        return 'EQ()'


class RQ(Kernel, Referentiable):
    """Rational quadratic kernel.

    Args:
        alpha (positive float): Shape of the prior over length scales.
            determines the weight of the tails of the kernel.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, alpha):
        self.alpha = alpha

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return (1 + .5 * B.pw_dists2(x, y) / self.alpha) ** (-self.alpha)

    def __str__(self):
        return 'RQ({})'.format(self.alpha)


class Exp(Kernel, PrimitiveType, Referentiable):
    """Exponential kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.exp(-B.pw_dists(x, y))

    def __str__(self):
        return 'Exp()'


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel, PrimitiveType, Referentiable):
    """Matern--3/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        r = 3 ** .5 * B.pw_dists(x, y)
        return (1 + r) * B.exp(-r)

    def __str__(self):
        return 'Matern32()'


class Matern52(Kernel, PrimitiveType, Referentiable):
    """Matern--5/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        r1 = 5 ** .5 * B.pw_dists(x, y)
        r2 = 5 * B.pw_dists2(x, y) / 3
        return (1 + r1 + r2) * B.exp(-r1)

    def __str__(self):
        return 'Matern52()'


class Delta(Kernel, PrimitiveType, Referentiable):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.cast(B.less(B.pw_dists2(x, y), self.epsilon), B.dtype(x))

    @property
    def length_scale(self):
        return 0

    def __str__(self):
        return 'Delta()'


class Linear(Kernel, PrimitiveType, Referentiable):
    """Linear kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.matmul(x, y, tr_b=True)

    @property
    def _stationary(self):
        return False

    @property
    def length_scale(self):
        return np.nan

    @property
    def var(self):
        return np.nan

    def __str__(self):
        return 'Linear()'


class PosteriorCrossKernel(Kernel, Referentiable):
    """Posterior cross kernel.

    Args:
        k_ij (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the left input and the right input respectively.
        k_zi (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the left input respectively.
        k_zj (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the right input respectively.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k_ij, k_zi, k_zj, z, Kz):
        self.k_ij = k_ij
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.Kz = Kz

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        qf = self.Kz.quadratic_form(self.k_zi(self.z, x, B),
                                    self.k_zj(self.z, y, B))
        return B.subtract(self.k_ij(x, y, B), qf)

    @property
    def _stationary(self):
        return False

    @property
    def length_scale(self):
        return np.nan

    @property
    def var(self):
        return np.nan

    def __str__(self):
        return 'PosteriorCrossKernel()'


class PosteriorKernel(PosteriorCrossKernel, Referentiable):
    """Posterior kernel.

    Args:
        gp (instance of :class:`.random.GP`): Prior GP.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, Kz):
        PosteriorCrossKernel.__init__(
            self, gp.kernel, gp.kernel, gp.kernel, z, Kz
        )

    def __str__(self):
        return 'PosteriorKernel()'


class DerivativeKernel(Kernel, DerivativeType, Referentiable):
    """Derivative of kernel."""
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        i, j = expand(self.derivs)
        k = self[0]

        # Derivative with respect to both `x` and `y`.
        if i is not None and j is not None:
            z = B.concat([x[:, i], y[:, j]], axis=0)
            n = B.shape(x)[0]
            K = k(B.concat([x[:, :i], z[:n, None], x[:, i + 1:]], axis=1),
                  B.concat([y[:, :j], z[n:, None], y[:, j + 1:]], axis=1)),
            return B.hessians(K, [z])[0][:n, n:]

        # Derivative with respect to `x`.
        elif j is None:
            xi = x[:, i:i + 1]
            # Give every `B.identity` a unique cache ID to prevent caching.
            xis = [B.identity(xi, cache_id=n) for n in range(B.shape_int(y)[0])]

            def f(z):
                return k(B.concat([x[:, :i], z[0], x[:, i + 1:]], axis=1), z[1])

            res = B.map_fn(f, (B.stack(xis, axis=0), y[:, None, :]),
                           dtype=B.dtype(x))
            return B.concat(B.gradients(B.sum(res, axis=0), xis), axis=1)

        # Derivative with respect to `y`.
        elif i is None:
            yj = y[:, j:j + 1]
            # Give every `B.identity` a unique cache ID to prevent caching.
            yjs = [B.identity(yj, cache_id=n) for n in range(B.shape_int(x)[0])]

            def f(z):
                return k(z[0], B.concat([y[:, :j], z[1], y[:, j + 1:]], axis=1))

            res = B.map_fn(f, (x[:, None, :], B.stack(yjs, axis=0)),
                           dtype=B.dtype(x))
            dKt = B.concat(B.gradients(B.sum(res, axis=0), yjs), axis=1)
            return B.transpose(dKt)

        else:
            raise RuntimeError('No derivative specified.')

    @property
    def _stationary(self):
        # NOTE: In the one-dimensional case, if derivatives with respect to both
        # arguments are taken, then the result is in fact stationary.
        return False

    @property
    def var(self):
        return np.nan

    @property
    def length_scale(self):
        return np.nan

    @property
    def period(self):
        return np.nan


class ReversedKernel(Kernel, WrappedType, Referentiable):
    """Kernel that evaluates with its arguments reversed."""
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.transpose(self[0](y, x, B))

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        return self[0].length_scale

    @property
    def period(self):
        return self[0].period

    def display(self, t):
        return 'Reversed({})'.format(t)


@dispatch.multi((Type, ReversedKernel),
                ({WrappedType, JoinType}, ReversedKernel))
def need_parens(el, parent): return False


@dispatch(ReversedKernel, ProductType)
def need_parens(el, parent): return False


# Periodicise kernels.

@dispatch(Kernel, object)
def periodicise(a, b): return PeriodicKernel(a, b)


@dispatch(ZeroKernel, object)
def periodicise(a, b): return a


# Reverse kernels.

@dispatch(Kernel)
def reverse(a): return a if a.stationary else ReversedKernel(a)


@dispatch(ReversedKernel)
def reverse(a): return a[0]


@dispatch.multi((ZeroKernel,), (OneKernel,))
def reverse(a): return a


# Propagate reversal.

@dispatch(SumKernel)
def reverse(a): return add(reverse(a[0]), reverse(a[1]))


@dispatch(ProductKernel)
def reverse(a): return mul(reverse(a[0]), reverse(a[1]))


@dispatch(ScaledKernel)
def reverse(a): return a.scale * reverse(a[0])


# Shifting:

@dispatch(Kernel, [object])
def shift(a, *shifts): return a if a.stationary else ShiftedKernel(a, *shifts)


@dispatch(ZeroKernel, [object])
def shift(a, *shifts): return a


@dispatch(ShiftedKernel, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))
