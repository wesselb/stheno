# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from numbers import Number
import logging

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable, PromisedType

from .input import Input

__all__ = ['Kernel', 'ProductKernel', 'SumKernel', 'ConstantKernel',
           'ScaledKernel', 'StretchedKernel', 'PeriodicKernel',
           'EQ', 'RQ', 'Matern12', 'Exp', 'Matern32', 'Matern52',
           'Delta', 'Linear', 'PosteriorKernel', 'ZeroKernel',
           'PosteriorCrossKernel', 'KernelCache', 'cache']

log = logging.getLogger(__name__)

dispatch = Dispatcher()


class KernelCache(Referentiable):
    """Cache for a kernel trace.

    Caches output of calls to kernels in `cache[kernel, x, y]`.

    Also caches calls to `B.*`: call instead `cache.*`.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self._kernel_outputs = {}
        self._other = {}

    @dispatch(object)
    def _resolve(self, key):
        return id(key)

    @dispatch({int, str, bool})
    def _resolve(self, key):
        return key

    @dispatch({tuple, list})
    def _resolve(self, key):
        return tuple(self._resolve(x) for x in key)

    def __getitem__(self, key):
        return self._kernel_outputs[self._resolve(key)]

    def __setitem__(self, key, output):
        self._kernel_outputs[self._resolve(key)] = output

    def __getattr__(self, f):
        def call_cached(*args, **kw_args):
            # Let the key depend on the function name...
            key = (f,)

            # ...on the arguments...
            key += self._resolve(args)

            # ...and on the keyword arguments.
            if len(kw_args) > 0:
                # First, sort keyword arguments according to keys.
                items = tuple(sorted(kw_args.items(), key=lambda x: x[0]))
                key += self._resolve(items)

            # Cached execution.
            if key in self._other:
                out = self._other[key]
                print('L2 hit!')
                return out
            else:
                self._other[key] = getattr(B, f)(*args, **kw_args)
                return self._other[key]

        return call_cached


def cache(f):
    """A decorator for `__call__` methods of kernels to cache their outputs."""

    def __call__(self, x, y, cache):
        try:
            out = cache[self, x, y]
            print('L1 hit!')
            return out
        except:
            pass

        # Try reverse of arguments.
        try:
            out = B.transpose(cache[self, y, x])
            print('L1 hit!')
            return out
        except KeyError:
            cache[self, x, y] = f(self, x, y, cache)
            return cache[self, x, y]

    return __call__


class Kernel(Referentiable):
    """Kernel function.

    Kernels can be added and multiplied.
    """
    # __metaclass__ = ABCMeta
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, KernelCache)
    def __call__(self, x, y, cache):
        """Construct the kernel for design matrices of points.

        Args:
            x (design matrix): First argument.
            y (design matrix, optional): Second argument. Defaults to first
                argument.
            cache (instance of :class:`.kernel.KernelCache`, optional): Cache.

        Returns:
            Kernel matrix.
        """
        raise NotImplementedError('Could not resolve kernel arguments.')

    @dispatch(object)
    def __call__(self, x):
        return self(x, x, KernelCache())

    @dispatch(object, KernelCache)
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self(x, y, KernelCache())

    @dispatch(Input)
    def __call__(self, x):
        return self(x, x, KernelCache())

    @dispatch(Input, KernelCache)
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(Input, Input)
    def __call__(self, x, y):
        return self(x, y, KernelCache())

    @dispatch(Input, Input, KernelCache)
    def __call__(self, x, y, cache):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get(), y.get(), cache)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __eq__(self, other):
        return equal(self, other)

    def stretch(self, amount):
        """Stretch the kernel.

        Args:
            amount (tensor): Stretch.
        """
        return stretch(self, amount)

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
        """Stationarity of the kernel"""
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

    @property
    def primitive(self):
        """Primitivess of the kernel. If a kernel is a primitive, then all
        instance of the kernel are considered equal.
        """
        return False


class ConstantKernel(Kernel, Referentiable):
    """Constant kernel of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return cache.ones((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return cache.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def length_scale(self):
        return np.inf

    def __str__(self):
        return '1'

    @property
    def primitive(self):
        return True


class ZeroKernel(Kernel, Referentiable):
    """Constant kernel of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return cache.zeros((1, 1), dtype=B.dtype(x))

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return cache.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def var(self):
        return 0

    @property
    def length_scale(self):
        return 0

    def __str__(self):
        return '0'

    @property
    def primitive(self):
        return True


class ScaledKernel(Kernel, Referentiable):
    """Scaled kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to scale.
        scale (tensor): Scale.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, scale):
        self.k = k
        self.scale = scale

    @dispatch(object, object, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return self.scale * self.k(x, y, cache)

    @property
    def _stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.scale * self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self.k.period

    def __str__(self):
        return '({} * {})'.format(self.scale, self.k)


class SumKernel(Kernel, Referentiable):
    """Sum of kernels.

    Args:
        k1 (instance of :class:`.kernel.Kernel`): First kernel in sum.
        k2 (instance of :class:`.kernel.Kernel`): Second kernel in sum.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @dispatch(object, object, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return self.k1(x, y, cache) + self.k2(x, y, cache)

    @property
    def _stationary(self):
        return self.k1.stationary and self.k2.stationary

    @property
    def var(self):
        return self.k1.var + self.k2.var

    @property
    def length_scale(self):
        return (self.k1.var * self.k1.length_scale +
                self.k2.var * self.k2.length_scale) / self.var

    @property
    def primitive(self):
        return False

    def __str__(self):
        return '({} + {})'.format(self.k1, self.k2)


class ProductKernel(Kernel, Referentiable):
    """Product of two kernels.

    Args:
        k1 (instance of :class:`.kernel.Kernel`): First kernel in product.
        k2 (instance of :class:`.kernel.Kernel`): Second kernel in product.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @dispatch(object, object, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return self.k1(x, y, cache) * self.k2(x, y, cache)

    @property
    def _stationary(self):
        return self.k1.stationary and self.k2.stationary

    @property
    def var(self):
        return self.k1.var * self.k2.var

    @property
    def length_scale(self):
        return B.minimum(self.k1.length_scale, self.k2.length_scale)

    def __str__(self):
        return '({} * {})'.format(self.k1, self.k2)


class StretchedKernel(Kernel, Referentiable):
    """Stretched kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to stretch.
        stretch (tensor): Stretch.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, stretch):
        self.k = k
        self._stretch = stretch

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return self.k(x / self._stretch, y / self._stretch, cache)

    @property
    def _stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale * self._stretch

    @property
    def period(self):
        return self.k.period * self._stretch

    def __str__(self):
        return '({} > {})'.format(self.k, self._stretch)


class PeriodicKernel(Kernel, Referentiable):
    """Periodic kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, period):
        self.k = k
        self._period = period

    @dispatch(Number, Number, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.array([[B.sin(z), B.cos(z)]])

        return self.k(feat_map(x), feat_map(y), cache)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.concat((B.sin(z), B.cos(z)), axis=1)

        return self.k(feat_map(x), feat_map(y), cache)

    @property
    def _stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self._period

    def __str__(self):
        return '({} per {})'.format(self.k, self._period)


class EQ(Kernel, Referentiable):
    """Exponentiated quadratic kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return B.exp(-.5 * cache.pw_dists2(x, y))

    def __str__(self):
        return 'EQ()'

    @property
    def primitive(self):
        return True


class RQ(Kernel, Referentiable):
    """Rational quadratic kernel.

    Args:
        alpha (positive float): Shape of the prior over length scales.
            determines the weight of the tails of the kernel.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, alpha):
        self.alpha = alpha

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return (1 + .5 * cache.pw_dists2(x, y) / self.alpha) ** (-self.alpha)

    def __str__(self):
        return 'RQ({})'.format(self.alpha)


class Exp(Kernel, Referentiable):
    """Exponential kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return B.exp(-cache.pw_dists(x, y))

    def __str__(self):
        return 'Exp()'

    @property
    def primitive(self):
        return True


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel, Referentiable):
    """Matern--3/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        r = 3 ** .5 * cache.pw_dists(x, y)
        return (1 + r) * B.exp(-r)

    def __str__(self):
        return 'Matern32()'

    @property
    def primitive(self):
        return True


class Matern52(Kernel, Referentiable):
    """Matern--5/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        r1 = 5 ** .5 * cache.pw_dists(x, y)
        r2 = 5 * cache.pw_dists2(x, y) / 3
        return (1 + r1 + r2) * B.exp(-r1)

    def __str__(self):
        return 'Matern52()'

    @property
    def primitive(self):
        return True


class Delta(Kernel, Referentiable):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    @dispatch(B.Numeric, B.Numeric, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        dists2 = cache.pw_dists2(x, y)
        return B.cast(dists2 < self.epsilon, B.dtype(x))

    @property
    def length_scale(self):
        return 0

    def __str__(self):
        return 'Delta()'

    @property
    def primitive(self):
        return True


class Linear(Kernel, Referentiable):
    """Linear kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, KernelCache)
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

    @property
    def primitive(self):
        return True


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

    @dispatch(object, object, KernelCache)
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
        return 'PosteriorCrossKernel(...)'


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


class ReversedKernel(Kernel, Referentiable):
    """Kernel that evaluates with its arguments reversed.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to evaluate.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k):
        self.k = k

    @dispatch(object, object, KernelCache)
    @cache
    def __call__(self, x, y, cache):
        return B.transpose(self.k(y, x, cache))

    @property
    def _stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self.k.period

    def __str__(self):
        return 'Reversed({})'.format(self.k)


# Generic multiplication.

@dispatch(object, Kernel)
def mul(a, b): return mul(b, a)


@dispatch(Kernel, object)
def mul(a, b):
    if b == 0:
        return ZeroKernel()
    elif b == 1:
        return a
    else:
        return ScaledKernel(a, b)


@dispatch(Kernel, Kernel)
def mul(a, b): return ProductKernel(a, b)


# Generic addition.

@dispatch(object, Kernel)
def add(a, b): return add(b, a)


@dispatch(Kernel, object)
def add(a, b):
    if b == 0:
        return a
    else:
        return SumKernel(a, mul(b, ConstantKernel()))


@dispatch(Kernel, Kernel)
def add(a, b): return SumKernel(a, b)


# Cancel redundant zeros and ones.

@dispatch.multi((ZeroKernel, object),
                (ZeroKernel, Kernel),
                (ZeroKernel, ZeroKernel),
                (Kernel, ConstantKernel),
                (ConstantKernel, ConstantKernel))
def mul(a, b): return a


@dispatch.multi((object, ZeroKernel),
                (Kernel, ZeroKernel),
                (ConstantKernel, Kernel))
def mul(a, b): return b


@dispatch.multi((ZeroKernel, object),
                (ZeroKernel, Kernel),
                (ZeroKernel, ZeroKernel))
def add(a, b): return b


@dispatch.multi((object, ZeroKernel),
                (Kernel, ZeroKernel))
def add(a, b): return a


# Group factors and terms if possible.

@dispatch(ScaledKernel, object)
def mul(a, b): return mul(a.scale * b, a.k)


@dispatch(object, ScaledKernel)
def mul(a, b): return mul(b.scale * a, b.k)


@dispatch(ScaledKernel, ScaledKernel)
def mul(a, b):
    if a.k == b.k:
        return ScaledKernel(a.k, a.scale * b.scale)
    else:
        return ProductKernel(a, b)


@dispatch(ScaledKernel, ScaledKernel)
def add(a, b):
    if a.k == b.k:
        return mul(a.scale + b.scale, a.k)
    else:
        return SumKernel(a, b)


# Distributive property:

@dispatch.multi((object, SumKernel),
                (Kernel, SumKernel))
def mul(a, b): return add(mul(a, b.k1), mul(a, b.k2))


@dispatch.multi((SumKernel, object),
                (SumKernel, Kernel))
def mul(a, b): return add(mul(a.k1, b), mul(a.k2, b))


@dispatch(SumKernel, SumKernel)
def mul(a, b):
    return add(add(mul(a.k1, b.k1), mul(a.k1, b.k2)),
               add(mul(a.k2, b.k1), mul(a.k2, b.k2)))


# Stretch kernels.

@dispatch(Kernel, object)
def stretch(a, b): return StretchedKernel(a, b)


@dispatch(ZeroKernel, object)
def stretch(a, b): return a


@dispatch(StretchedKernel, object)
def stretch(a, b): return stretch(a.k, a._stretch * b)


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
def reverse(a): return a.k


@dispatch.multi((ZeroKernel,), (ConstantKernel,))
def reverse(a): return a


# Propagate reversal.

@dispatch(SumKernel)
def reverse(a): return add(reverse(a.k1), reverse(a.k2))


@dispatch(ProductKernel)
def reverse(a): return mul(reverse(a.k1), reverse(a.k2))


@dispatch(ScaledKernel)
def reverse(a): return a.scale * reverse(a.k)


# Equality:

@dispatch(object, object)
def equal(a, b): return False


@dispatch(Kernel, Kernel)
def equal(a, b): return a.primitive and b.primitive and type(a) == type(b)


@dispatch.multi((SumKernel, SumKernel),
                (ProductKernel, ProductKernel))
def equal(a, b): return (equal(a.k1, b.k1) and equal(a.k2, b.k2)) or \
                        (equal(a.k1, b.k2) and equal(a.k2, b.k1))
