# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Self, Referentiable
from stheno.function_field import StretchedFunction, ShiftedFunction, \
    SelectedFunction, InputTransformedFunction, DerivativeFunction, \
    TensorProductFunction, ZeroFunction, OneFunction, \
    ScaledFunction, ProductFunction, SumFunction, Function

from .cache import Cache, cache, uprank
from .field import apply_optional_arg, get_field
from .input import Input

__all__ = ['TensorProductMean', 'DerivativeMean']

log = logging.getLogger(__name__)


class Mean(Function, Referentiable):
    """Mean function.

    Means can be added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    def __call__(self, x, cache):
        """Construct the mean for a design matrix.

        Args:
            x (input): Points to construct the mean at.
            cache (:class:`.cache.Cache`): Cache.

        Returns:
            tensor: Mean vector as a rank 2 column vector.
        """
        raise NotImplementedError()

    @_dispatch(object)
    def __call__(self, x):
        return self(x, Cache())

    @_dispatch(Input)
    def __call__(self, x):
        return self(x, Cache())

    @_dispatch(Input, Cache)
    def __call__(self, x, cache):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get(), cache)


# Register the field.
@get_field.extend(Mean)
def get_field(a): return Mean


class SumMean(Mean, SumFunction, Referentiable):
    """Sum of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.add(self[0](x, B), self[1](x, B))


class ProductMean(Mean, ProductFunction, Referentiable):
    """Product of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.multiply(self[0](x, B), self[1](x, B))


class ScaledMean(Mean, ScaledFunction, Referentiable):
    """Scaled mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.multiply(self.scale, self[0](x, B))


class StretchedMean(Mean, StretchedFunction, Referentiable):
    """Stretched mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.divide(x, self.stretches[0]), B)


class ShiftedMean(Mean, ShiftedFunction, Referentiable):
    """Shifted mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.subtract(x, self.shifts[0]), B)


class SelectedMean(Mean, SelectedFunction, Referentiable):
    """Mean with particular input dimensions selected."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.take(x, self.dims[0], axis=1), B)


class InputTransformedMean(Mean, InputTransformedFunction, Referentiable):
    """Input-transformed mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](apply_optional_arg(self.fs[0], x, B), B)


class OneMean(Mean, OneFunction, Referentiable):
    """Constant mean of `1`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return B.ones([B.shape(x)[0], 1], dtype=B.dtype(x))


class ZeroMean(Mean, ZeroFunction, Referentiable):
    """Constant mean of `0`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return B.zeros([B.shape(x)[0], 1], dtype=B.dtype(x))


class TensorProductMean(Mean, TensorProductFunction, Referentiable):
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return apply_optional_arg(self.fs[0], x, B)


class DerivativeMean(Mean, DerivativeFunction, Referentiable):
    """Derivative of mean."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        i = self.derivs[0]
        return B.gradients(self[0](x, B), x)[0][:, i:i + 1]


class PosteriorCrossMean(Mean, Referentiable):
    """Posterior cross mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to
            the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to
            the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the input respectively.
        z (input): Locations of data.
        K_z (:class:`.matrix.Dense`): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, m_i, m_z, k_zi, z, K_z, y):
        self.m_i = m_i
        self.k_zi = k_zi
        self.m_z = m_z
        self.z = z
        self.K_z = K_z
        self.y = uprank(y)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        diff = B.subtract(self.y, self.m_z(self.z, B))
        return B.add(self.m_i(x, B),
                     B.qf(self.K_z, self.k_zi(self.z, x), diff))


class PosteriorMean(PosteriorCrossMean, Referentiable):
    """Posterior mean.

    Args:
        gp (:class:`.random.GP`): Corresponding GP.
        z (input): Locations of data.
        K_z (:class:`.matrix.Dense`): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, K_z, y):
        PosteriorCrossMean.__init__(self, gp.mean, gp.mean, gp.kernel,
                                    z, K_z, y)


class VariationalPosteriorCrossMean(Mean, Referentiable):
    """Variational posterior cross mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to
            the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to
            the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the input respectively.
        z (input): Locations of pseudo-points.
        mu (tensor): Variational mean of the pseudo-points premultiplied by the
            prior covariance of the pseudo-points.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, m_i, m_z, k_zi, z, mu):
        self.m_i = m_i
        self.k_zi = k_zi
        self.m_z = m_z
        self.z = z
        self.mu = uprank(mu)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        diff = B.subtract(self.mu, self.m_z(self.z, B))
        K_zi = self.k_zi(self.z, x)
        return B.add(self.m_i(x, B), B.matmul(K_zi, diff, tr_a=True))
