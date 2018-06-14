# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import operator

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

from .cache import cache, Cache, uprank
from .field import add, mul, dispatch, Type, PrimitiveType, \
    ZeroType, OneType, ScaledType, ProductType, SumType, StretchedType, \
    WrappedType, ShiftedType, SelectedType, InputTransformedType, JoinType, \
    DerivativeType, broadcast, shift, stretch, select, transform, \
    FunctionType, apply_optional_arg
from .input import Input

__all__ = ['Kernel', 'OneKernel', 'ZeroKernel', 'ScaledKernel', 'EQ', 'RQ',
           'Matern12', 'Exp', 'Matern32', 'Matern52', 'Delta', 'Linear',
           'DerivativeKernel']

log = logging.getLogger(__name__)


def expand(xs):
    """Expand a sequence to the same element repeated twice if there is only
    one element.

    Args:
        xs (sequence): Sequence to expand.
    """
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
        raise RuntimeError('For kernel "{}", could not resolve '
                           'arguments "{}" and "{}".'.format(self, x, y))

    @dispatch(object)
    def __call__(self, x):
        return self(x, x, Cache())

    @dispatch(object, Cache)
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @dispatch(Input, Input)
    def __call__(self, x, y):
        return self(x, y, Cache())

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
        return False

    @property
    def var(self):
        """Variance of the kernel"""
        raise RuntimeError('The variance of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))

    @property
    def length_scale(self):
        """Approximation to the length scale of the kernel. Returns `np.nan`
        if the length scale is undefined or cannot be determined.
        """
        raise RuntimeError('The length scale of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))

    @property
    def period(self):
        """Period of the kernel. Returns `np.nan` is the period is undefined
        or cannot be determined.
        """
        raise RuntimeError('The period of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))


class OneKernel(Kernel, OneType, Referentiable):
    """Constant kernel of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return B.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 0

    @property
    def period(self):
        return 0


class ZeroKernel(Kernel, ZeroType, Referentiable):
    """Constant kernel of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return B.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 0

    @property
    def length_scale(self):
        return 0

    @property
    def period(self):
        return 0


class ScaledKernel(Kernel, ScaledType, Referentiable):
    """Scaled kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.multiply(self.scale, self[0](x, y, B))

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

    @property
    def period(self):
        return np.inf


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

    @property
    def period(self):
        return np.inf


class StretchedKernel(Kernel, StretchedType, Referentiable):
    """Stretched kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        stretches1, stretches2 = expand(self.stretches)
        return self[0](B.divide(x, stretches1), B.divide(y, stretches2), B)

    @property
    def _stationary(self):
        if len(self.stretches) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        if len(self.stretches) == 1:
            return self[0].length_scale * self.stretches[0]
        else:
            # NOTE: Can do something more clever here.
            return Kernel.length_scale.fget(self)

    @property
    def period(self):
        if len(self.stretches) == 1:
            return self[0].period * self.stretches[0]
        else:
            # NOTE: Can do something more clever here.
            return Kernel.period.fget(self)


class ShiftedKernel(Kernel, ShiftedType, Referentiable):
    """Shifted kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        shifts1, shifts2 = expand(self.shifts)
        return self[0](B.subtract(x, shifts1), B.subtract(y, shifts2), B)

    @property
    def _stationary(self):
        if len(self.shifts) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @property
    def var(self):
        return self[0].var

    @property
    def length_scale(self):
        if len(self.shifts) == 1:
            return self[0].length_scale
        else:
            # NOTE: Can do something more clever here.
            return Kernel.length_scale.fget(self)

    @property
    def period(self):
        return self[0].period


class SelectedKernel(Kernel, SelectedType, Referentiable):
    """Kernel with particular input dimensions selected."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        dims1, dims2 = expand(self.dims)
        x = x if dims1 is None else B.take(x, dims1, axis=1)
        y = y if dims2 is None else B.take(y, dims2, axis=1)
        return self[0](x, y, B)

    @property
    def _stationary(self):
        if len(self.dims) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

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
                # NOTE: Can do something more clever here.
                return Kernel.length_scale.fget(self)

    @property
    def period(self):
        period = self[0].period
        if B.is_scalar(period):
            return period
        else:
            if len(self.dims) == 1:
                return B.take(period, self.dims[0])
            else:
                # NOTE: Can do something more clever here.
                return Kernel.period.fget(self)


class InputTransformedKernel(Kernel, InputTransformedType, Referentiable):
    """Input-transformed kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        f1, f2 = expand(self.fs)
        x = x if f1 is None else apply_optional_arg(f1, x, B)
        y = y if f2 is None else apply_optional_arg(f2, y, B)
        return self[0](x, y, B)


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

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
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
    @uprank
    def __call__(self, x, y, B):
        return B.exp(B.multiply(B.cast(-.5, dtype=B.dtype(x)),
                                B.pw_dists2(x, y)))

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 1

    @property
    def period(self):
        return np.inf


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
    @uprank
    def __call__(self, x, y, B):
        return (1 + .5 * B.pw_dists2(x, y) / self.alpha) ** (-self.alpha)

    def __str__(self):
        return 'RQ({})'.format(self.alpha)

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 1

    @property
    def period(self):
        return np.inf


class Exp(Kernel, PrimitiveType, Referentiable):
    """Exponential kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return B.exp(-B.pw_dists(x, y))

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 1

    @property
    def period(self):
        return np.inf


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel, PrimitiveType, Referentiable):
    """Matern--3/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        r = 3 ** .5 * B.pw_dists(x, y)
        return (1 + r) * B.exp(-r)

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 1

    @property
    def period(self):
        return np.inf


class Matern52(Kernel, PrimitiveType, Referentiable):
    """Matern--5/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        r1 = 5 ** .5 * B.pw_dists(x, y)
        r2 = 5 * B.pw_dists2(x, y) / 3
        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 1

    @property
    def period(self):
        return np.inf


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
    @uprank
    def __call__(self, x, y, B):
        return B.cast(B.less(B.pw_dists2(x, y), self.epsilon), B.dtype(x))

    @property
    def _stationary(self):
        return True

    @property
    def var(self):
        return 1

    @property
    def length_scale(self):
        return 0

    @property
    def period(self):
        return np.inf


class Linear(Kernel, PrimitiveType, Referentiable):
    """Linear kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return B.matmul(x, y, tr_b=True)

    @property
    def _stationary(self):
        return False

    @property
    def period(self):
        return np.inf


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


class DerivativeKernel(Kernel, DerivativeType, Referentiable):
    """Derivative of kernel."""
    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
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
        elif i is not None and j is None:
            xi = x[:, i:i + 1]
            # Give every `B.identity` a unique cache ID to prevent caching.
            xis = [B.identity(xi, cache_id=n) for n in range(B.shape_int(y)[0])]

            def f(z):
                return k(B.concat([x[:, :i], z[0], x[:, i + 1:]], axis=1), z[1])

            res = B.map_fn(f, (B.stack(xis, axis=0), y[:, None, :]),
                           dtype=B.dtype(x))
            return B.concat(B.gradients(B.sum(res, axis=0), xis), axis=1)

        # Derivative with respect to `y`.
        elif i is None and j is not None:
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


class FunctionKernel(Kernel, FunctionType, Referentiable):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        f1, f2 = expand(self.fs)
        return B.multiply(apply_optional_arg(f1, x, B),
                          B.transpose(apply_optional_arg(f2, y, B)))


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


@dispatch(ShiftedKernel)
def reverse(a): return shift(reversed(a[0]), *reversed(a.shifts))


@dispatch(StretchedKernel)
def reverse(a): return stretch(reversed(a[0]), *reversed(a.stretches))


@dispatch(InputTransformedKernel)
def reverse(a): return transform(reversed(a[0]), *reversed(a.fs))


@dispatch(SelectedKernel)
def reverse(a): return select(reversed(a[0]), *reversed(a.dims))


# Propagate reversal.

@dispatch(SumKernel)
def reverse(a): return add(reverse(a[0]), reverse(a[1]))


@dispatch(ProductKernel)
def reverse(a): return mul(reverse(a[0]), reverse(a[1]))


@dispatch(ScaledKernel)
def reverse(a): return a.scale * reverse(a[0])


# Shifting:

@dispatch(Kernel, [object])
def shift(a, *shifts):
    if a.stationary and len(shifts) == 1:
        return a
    else:
        return ShiftedKernel(a, *shifts)


@dispatch(ZeroKernel, [object])
def shift(a, *shifts): return a


@dispatch(ShiftedKernel, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))
