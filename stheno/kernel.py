# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from numbers import Number

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

from .cache import cache, Cache
from .field import add, mul, dispatch, Type, PrimitiveType, \
    ZeroType, OneType, ScaledType, ProductType, SumType, StretchedType, \
    WrappedType
from .input import Input

__all__ = ['ScaledKernel', 'EQ', 'RQ', 'Matern12', 'Exp', 'Matern32',
           'Matern52', 'Delta', 'Linear']

log = logging.getLogger(__name__)


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

    @dispatch(object, Cache)
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @dispatch(Input)
    def __call__(self, x):
        return self(x, x, Cache())

    @dispatch(Input, Cache)
    def __call__(self, x, cache):
        return self(x, x, cache)

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
            period (tensor): Period. Defaults to `1`.
        """
        return periodicise(self, period)

    def __reversed__(self):
        """Reverse the arguments of the kernel."""
        return reverse(self)

    @property
    def stationary(self):
        """Stationarity of the kernel"""
        if hasattr(self, '_stationary_cache'):
            return self._stationary_cache
        else:
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
    def __call__(self, x, y, cache):
        return cache.ones((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        return cache.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def length_scale(self):
        return np.inf


class ZeroKernel(Kernel, ZeroType, Referentiable):
    """Constant kernel of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number, Cache)
    @cache
    def __call__(self, x, y, cache):
        return cache.zeros((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        return cache.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

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
    def __call__(self, x, y, cache):
        return self.scale * self[0](x, y, cache)

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
    def __call__(self, x, y, cache):
        return self[0](x, y, cache) + self[1](x, y, cache)

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
    def __call__(self, x, y, cache):
        return self[0](x, y, cache) * self[1](x, y, cache)

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
    def __call__(self, x, y, cache):
        return self[0](x / self.extent, y / self.extent, cache)

    @property
    def _stationary(self):
        return self[0].stationary

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        return self[0].length_scale * self.extent

    @property
    def period(self):
        return self[0].period * self.extent


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
    def __call__(self, x, y, cache):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.array([[B.sin(z), B.cos(z)]])

        return self[0](feat_map(x), feat_map(y), cache)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.concat((B.sin(z), B.cos(z)), axis=1)

        return self[0](feat_map(x), feat_map(y), cache)

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
    def __call__(self, x, y, cache):
        return B.exp(-.5 * cache.pw_dists2(x, y))

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
    def __call__(self, x, y, cache):
        return (1 + .5 * cache.pw_dists2(x, y) / self.alpha) ** (-self.alpha)

    def __str__(self):
        return 'RQ({})'.format(self.alpha)


class Exp(Kernel, PrimitiveType, Referentiable):
    """Exponential kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        return B.exp(-cache.pw_dists(x, y))

    def __str__(self):
        return 'Exp()'


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel, PrimitiveType, Referentiable):
    """Matern--3/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        r = 3 ** .5 * cache.pw_dists(x, y)
        return (1 + r) * B.exp(-r)

    def __str__(self):
        return 'Matern32()'


class Matern52(Kernel, PrimitiveType, Referentiable):
    """Matern--5/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, cache):
        r1 = 5 ** .5 * cache.pw_dists(x, y)
        r2 = 5 * cache.pw_dists2(x, y) / 3
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
    def __call__(self, x, y, cache):
        dists2 = cache.pw_dists2(x, y)
        return B.cast(dists2 < self.epsilon, B.dtype(x))

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
    def __call__(self, x, y, cache):
        return cache.matmul(x, y, tr_b=True)

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
    def __call__(self, x, y, cache):
        return (self.k_ij(x, y, cache) -
                self.Kz.quadratic_form(self.k_zi(self.z, x, cache),
                                       self.k_zj(self.z, y, cache)))

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


class ReversedKernel(Kernel, WrappedType, Referentiable):
    """Kernel that evaluates with its arguments reversed."""
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, cache):
        return B.transpose(self[0](y, x, cache))

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


@dispatch(Type, ReversedKernel)
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
def reverse(a):
    if a.stationary:
        return a
    else:
        return ReversedKernel(a)


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
