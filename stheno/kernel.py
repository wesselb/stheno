# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import operator

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

from stheno.function_field import StretchedFunction, ShiftedFunction, \
    SelectedFunction, InputTransformedFunction, DerivativeFunction, \
    TensorProductFunction, stretch, transform, Function, ZeroFunction, \
    OneFunction, ScaledFunction, ProductFunction, SumFunction, \
    WrappedFunction, PrimitiveFunction, JoinFunction, shift, select, \
    to_tensor
from .cache import cache, Cache, uprank
from .field import add, mul, broadcast, apply_optional_arg, get_field, \
    Formatter, need_parens
from .input import Input
from .spd import SPD, Diagonal, LowRank, UniformDiagonal, OneSPD, ZeroSPD, \
    dense

__all__ = ['Kernel', 'OneKernel', 'ZeroKernel', 'ScaledKernel', 'EQ', 'RQ',
           'Matern12', 'Exp', 'Matern32', 'Matern52', 'Delta', 'Linear',
           'DerivativeKernel']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


def expand(xs):
    """Expand a sequence to the same element repeated twice if there is only
    one element.

    Args:
        xs (sequence): Sequence to expand.

    Returns:
        object: `xs * 2` or `xs`.
    """
    return xs * 2 if len(xs) == 1 else xs


class Kernel(Function, Referentiable):
    """Kernel function.

    Kernels can be added and multiplied.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    def __call__(self, x, y, cache):
        """Construct the kernel matrix between all `x` and `y`.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first
                argument.
            cache (:class:`.cache.Cache`, optional): Cache.

        Returns:
            :class:`.spd.SPD:: Kernel matrix.
        """
        raise RuntimeError('For kernel "{}", could not resolve '
                           'arguments "{}" and "{}".'.format(self, x, y))

    @_dispatch(object)
    def __call__(self, x):
        return self(x, x, Cache())

    @_dispatch(object, Cache)
    def __call__(self, x, cache):
        return self(x, x, cache)

    @_dispatch(object, object)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @_dispatch(Input, Input)
    def __call__(self, x, y):
        return self(x, y, Cache())

    @_dispatch(Input, Input, Cache)
    def __call__(self, x, y, cache):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get(), y.get(), cache)

    @_dispatch(object, object, Cache)
    def elwise(self, x, y, cache):
        """Construct the kernel vector `x` and `y` element-wise.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first
                argument.
            cache (:class:`.cache.Cache`, optional): Cache.

        Returns:
            tensor: Kernel vector as a rank 2 column vector.
        """
        return B.expand_dims(B.diag(self(x, y, cache)), 1)

    @_dispatch(object)
    def elwise(self, x):
        return self.elwise(x, x, Cache())

    @_dispatch(object, Cache)
    def elwise(self, x, cache):
        return self.elwise(x, x, cache)

    @_dispatch(object, object)
    def elwise(self, x, y):
        return self.elwise(x, y, Cache())

    @_dispatch(Input, Input)
    def elwise(self, x, y):
        return self.elwise(x, y, Cache())

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor, optional): Period. Defaults to `1`.

        Returns:
            :class:`.kernel.Kernel`: Periodic version of the kernel.
        """
        return periodicise(self, to_tensor(period))

    def __reversed__(self):
        """Reverse the arguments of the kernel."""
        return reverse(self)

    @property
    def stationary(self):
        """Stationarity of the kernel."""
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
        """Variance of the kernel."""
        raise RuntimeError('The variance of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))

    @property
    def length_scale(self):
        """Approximation of the length scale of the kernel."""
        raise RuntimeError('The length scale of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))

    @property
    def period(self):
        """Period of the kernel."""
        raise RuntimeError('The period of "{}" could not be determined.'
                           ''.format(self.__class__.__name__))


# Register the field.
@get_field.extend(Kernel)
def get_field(a): return Kernel


class OneKernel(Kernel, OneFunction, Referentiable):
    """Constant kernel of `1`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = B.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))
        return OneSPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return B.ones((B.shape(x)[0], 1), dtype=B.dtype(x))

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


class ZeroKernel(Kernel, ZeroFunction, Referentiable):
    """Constant kernel of `0`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = B.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))
        return ZeroSPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return B.zeros((B.shape(x)[0], 1), dtype=B.dtype(x))

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


class ScaledKernel(Kernel, ScaledFunction, Referentiable):
    """Scaled kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return self._compute(self[0](x, y, B), B)

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        return self._compute(self[0].elwise(x, y, B), B)

    def _compute(self, K, B):
        return B.multiply(B.cast(self.scale, dtype=B.dtype(K)), K)

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


class SumKernel(Kernel, SumFunction, Referentiable):
    """Sum of kernels."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.add(self[0](x, y, B), self[1](x, y, B))

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        return B.add(self[0].elwise(x, y, B), self[1].elwise(x, y, B))

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


class ProductKernel(Kernel, ProductFunction, Referentiable):
    """Product of two kernels."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.multiply(self[0](x, y, B), self[1](x, y, B))

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        return B.multiply(self[0].elwise(x, y, B), self[1].elwise(x, y, B))

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


class StretchedKernel(Kernel, StretchedFunction, Referentiable):
    """Stretched kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return self[0](*self._compute(x, y, B))

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self[0].elwise(*self._compute(x, y, B))

    def _compute(self, x, y, B):
        stretches1, stretches2 = expand(self.stretches)
        return B.divide(x, stretches1), B.divide(y, stretches2), B

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


class ShiftedKernel(Kernel, ShiftedFunction, Referentiable):
    """Shifted kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return self[0](*self._compute(x, y, B))

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self[0].elwise(*self._compute(x, y, B))

    def _compute(self, x, y, B):
        shifts1, shifts2 = expand(self.shifts)
        return B.subtract(x, shifts1), B.subtract(y, shifts2), B

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


class SelectedKernel(Kernel, SelectedFunction, Referentiable):
    """Kernel with particular input dimensions selected."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return self[0](*self._compute(x, y, B))

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self[0].elwise(*self._compute(x, y, B))

    def _compute(self, x, y, B):
        dims1, dims2 = expand(self.dims)
        x = x if dims1 is None else B.take(x, dims1, axis=1)
        y = y if dims2 is None else B.take(y, dims2, axis=1)
        return x, y, B

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


class InputTransformedKernel(Kernel, InputTransformedFunction, Referentiable):
    """Input-transformed kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return self[0](*self._compute(x, y, B))

    @_dispatch(object, object, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self[0].elwise(*self._compute(x, y, B))

    def _compute(self, x, y, B):
        f1, f2 = expand(self.fs)
        x = x if f1 is None else apply_optional_arg(f1, x, B)
        y = y if f2 is None else apply_optional_arg(f2, y, B)
        return x, y, B


class PeriodicKernel(Kernel, WrappedFunction, Referentiable):
    """Periodic kernel.

    Args:
        k (:class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, period):
        WrappedFunction.__init__(self, k)
        self._period = period

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return self[0](*self._compute(x, y, B))

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self[0].elwise(*self._compute(x, y, B))

    def _compute(self, x, y, B):
        def feat_map(z):
            z = B.divide(B.multiply(B.multiply(z, 2), B.pi), self.period)
            return B.concat((B.sin(z), B.cos(z)), axis=1)

        return feat_map(x), feat_map(y), B

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

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        return '{} per {}'.format(e, formatter(self._period))


class EQ(Kernel, PrimitiveFunction, Referentiable):
    """Exponentiated quadratic kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = self._compute(B.pw_dists2(x, y), B)
        return SPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self._compute(B.ew_dists2(x, y), B)

    def _compute(self, dists2, B):
        return B.exp(B.multiply(B.cast(-.5, dtype=B.dtype(dists2)), dists2))

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
        alpha (scalar): Shape of the prior over length scales. Determines the
            weight of the tails of the kernel. Must be positive.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, alpha):
        self.alpha = alpha

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = self._compute(B.pw_dists2(x, y), B)
        return SPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self._compute(B.ew_dists2(x, y), B)

    def _compute(self, dists2, B):
        return (1 + .5 * dists2 / self.alpha) ** (-self.alpha)

    @_dispatch(Formatter)
    def display(self, formatter):
        return 'RQ({})'.format(formatter(self.alpha))

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


class Exp(Kernel, PrimitiveFunction, Referentiable):
    """Exponential kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = B.exp(-B.pw_dists(x, y))
        return SPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return B.exp(-B.ew_dists(x, y))

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


class Matern32(Kernel, PrimitiveFunction, Referentiable):
    """Matern--3/2 kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = self._compute(B.pw_dists(x, y), B)
        return SPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self._compute(B.ew_dists(x, y), B)

    def _compute(self, dists, B):
        r = 3 ** .5 * dists
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


class Matern52(Kernel, PrimitiveFunction, Referentiable):
    """Matern--5/2 kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        out = self._compute(B.pw_dists(x, y), B)
        return SPD(out) if x is y else out

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self._compute(B.ew_dists(x, y), B)

    def _compute(self, dists, B):
        r1 = 5 ** .5 * dists
        r2 = 5 * dists ** 2 / 3
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


class Delta(Kernel, PrimitiveFunction, Referentiable):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
            Defaults to `1e-6`.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        if x is y:
            return UniformDiagonal(B.cast(1, dtype=B.dtype(x)),
                                   B.shape(x)[0])
        else:
            return self._compute(B.pw_dists2(x, y), B)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return self._compute(B.ew_dists2(x, y), B)

    def _compute(self, dists2, B):
        return B.cast(B.less(dists2, self.epsilon), B.dtype(dists2))

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


class Linear(Kernel, PrimitiveFunction, Referentiable):
    """Linear kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        return LowRank(x) if x is y else B.matmul(x, y, tr_b=True)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        return B.expand_dims(B.sum(B.multiply(x, y), axis=1), 1)

    @property
    def _stationary(self):
        return False

    @property
    def period(self):
        return np.inf


class PosteriorCrossKernel(Kernel, Referentiable):
    """Posterior cross kernel.

    Args:
        k_ij (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the left input and the right input respectively.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the left input respectively.
        k_zj (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the right input respectively.
        z (input): Locations of data.
        K_z (:class:`.spd.SPD`): Kernel matrix of data.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, k_ij, k_zi, k_zj, z, K_z):
        self.k_ij = k_ij
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.K_z = K_z

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        qf = self.K_z.quadratic_form(self.k_zi(self.z, x, B),
                                     self.k_zj(self.z, y, B))
        return B.subtract(self.k_ij(x, y, B), qf)

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        qf_diag = self.K_z.quadratic_form_diag(self.k_zi(self.z, x, B),
                                               self.k_zj(self.z, y, B))
        return B.subtract(self.k_ij.elwise(x, y, B), B.expand_dims(qf_diag, 1))


class PosteriorKernel(PosteriorCrossKernel, Referentiable):
    """Posterior kernel.

    Args:
        gp (:class:`.random.GP`): Prior GP.
        z (input): Locations of data.
        Kz (:class:`.spd.SPD`): Kernel matrix of data.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, Kz):
        PosteriorCrossKernel.__init__(
            self, gp.kernel, gp.kernel, gp.kernel, z, Kz
        )


class VariationalPosteriorCrossKernel(Kernel, Referentiable):
    """Variational posterior cross kernel.

    Args:
        k_ij (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the left input and the right input respectively.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the left input respectively.
        k_zj (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the right input respectively.
        z (input): Locations of the pseudo-points.
        Kz (:class:`.spd.SPD`): Prior covariance of the pseudo-points.
        A (:class:`.spd.SPD): Variational covariance of the pseudo-points
            premultiplied and postmultiplied by `Kz`.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, k_ij, k_zi, k_zj, z, K_z, A):
        self.k_ij = k_ij
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.K_z = K_z
        self.A = A

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        K_zi = self.k_zi(self.z, x, B)
        K_zj = self.k_zj(self.z, y, B)
        qf1 = self.K_z.quadratic_form(K_zi, K_zj)
        qf2 = self.A.quadratic_form(K_zi, K_zj)
        return B.add(B.subtract(self.k_ij(x, y, B), qf1), qf2)

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        K_zi = self.k_zi(self.z, x, B)
        K_zj = self.k_zj(self.z, y, B)
        qf1_diag = B.expand_dims(self.K_z.quadratic_form_diag(K_zi, K_zj), 1)
        qf2_diag = B.expand_dims(self.A.quadratic_form_diag(K_zi, K_zj), 1)
        return B.add(B.subtract(self.k_ij.elwise(x, y, B), qf1_diag), qf2_diag)


class DerivativeKernel(Kernel, DerivativeFunction, Referentiable):
    """Derivative of kernel."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        i, j = expand(self.derivs)
        k = self[0]

        # Derivative with respect to both `x` and `y`.
        if i is not None and j is not None:
            z = B.concat([x[:, i], y[:, j]], axis=0)
            n = B.shape(x)[0]
            K = dense(
                k(B.concat([x[:, :i], z[:n, None], x[:, i + 1:]], axis=1),
                  B.concat([y[:, :j], z[n:, None], y[:, j + 1:]], axis=1))
            )
            return B.hessians(K, [z])[0][:n, n:]

        # Derivative with respect to `x`.
        elif i is not None and j is None:
            xi = x[:, i:i + 1]
            # Give every `B.identity` a unique cache ID to prevent caching.
            xis = [B.identity(xi, cache_id=n) for n in range(B.shape_int(y)[0])]

            def f(z):
                return dense(
                    k(B.concat([x[:, :i], z[0], x[:, i + 1:]], axis=1), z[1])
                )

            res = B.map_fn(f, (B.stack(xis, axis=0), y[:, None, :]),
                           dtype=B.dtype(x))
            return B.concat(B.gradients(B.sum(res, axis=0), xis), axis=1)

        # Derivative with respect to `y`.
        elif i is None and j is not None:
            yj = y[:, j:j + 1]
            # Give every `B.identity` a unique cache ID to prevent caching.
            yjs = [B.identity(yj, cache_id=n) for n in range(B.shape_int(x)[0])]

            def f(z):
                return dense(
                    k(z[0], B.concat([y[:, :j], z[1], y[:, j + 1:]], axis=1))
                )

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


class TensorProductKernel(Kernel, TensorProductFunction, Referentiable):
    """Tensor product kernel."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, y, B):
        f1, f2 = expand(self.fs)
        if x is y and f1 is f2:
            return LowRank(apply_optional_arg(f1, x, B))
        else:
            return B.multiply(apply_optional_arg(f1, x, B),
                              B.transpose(apply_optional_arg(f2, y, B)))

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    @uprank
    def elwise(self, x, y, B):
        f1, f2 = expand(self.fs)
        return B.multiply(apply_optional_arg(f1, x, B),
                          apply_optional_arg(f2, y, B))


class ReversedKernel(Kernel, WrappedFunction, Referentiable):
    """Reversed kernel.

    Evaluates with its arguments reversed.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.transpose(self[0](y, x, B))

    @_dispatch(object, object, Cache)
    @cache
    def elwise(self, x, y, B):
        return self[0].elwise(y, x, B)

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

    @_dispatch(object, Formatter)
    def display(self, e, formatter):
        return 'Reversed({})'.format(e)


@need_parens.extend_multi((Function, ReversedKernel),
                          ({WrappedFunction, JoinFunction}, ReversedKernel))
def need_parens(el, parent): return False


@need_parens.extend(ReversedKernel, ProductFunction)
def need_parens(el, parent): return False


# Periodicise kernels.

@_dispatch(Kernel, object)
def periodicise(a, b): return PeriodicKernel(a, b)


@_dispatch(ZeroKernel, object)
def periodicise(a, b): return a


# Reverse kernels.

@_dispatch(Kernel)
def reverse(a): return a if a.stationary else ReversedKernel(a)


@_dispatch(ReversedKernel)
def reverse(a): return a[0]


@_dispatch.multi((ZeroKernel,), (OneKernel,))
def reverse(a): return a


@_dispatch(ShiftedKernel)
def reverse(a): return shift(reversed(a[0]), *reversed(a.shifts))


@_dispatch(StretchedKernel)
def reverse(a): return stretch(reversed(a[0]), *reversed(a.stretches))


@_dispatch(InputTransformedKernel)
def reverse(a): return transform(reversed(a[0]), *reversed(a.fs))


@_dispatch(SelectedKernel)
def reverse(a): return select(reversed(a[0]), *reversed(a.dims))


# Propagate reversal.

@_dispatch(SumKernel)
def reverse(a): return add(reverse(a[0]), reverse(a[1]))


@_dispatch(ProductKernel)
def reverse(a): return mul(reverse(a[0]), reverse(a[1]))


@_dispatch(ScaledKernel)
def reverse(a): return mul(a.scale, reversed(a[0]))


# Make shifting synergise with reversal.

@shift.extend(Kernel, [object])
def shift(a, *shifts):
    if a.stationary and len(shifts) == 1:
        return a
    else:
        return ShiftedKernel(a, *shifts)


@shift.extend(ZeroKernel, [object])
def shift(a, *shifts): return a


@shift.extend(ShiftedKernel, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))
