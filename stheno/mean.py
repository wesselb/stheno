import logging

import algebra as algebra
from lab import B
from matrix import AbstractMatrix
from plum import Dispatcher, convert, Union

from . import PromisedFDD as FDD
from .input import Input, MultiInput
from .kernel import unwrap
from .util import num_elements, uprank

__all__ = ["TensorProductMean", "DerivativeMean"]

_dispatch = Dispatcher()

log = logging.getLogger(__name__)


class Mean(algebra.Function):
    """Mean function.

    Means can be added and multiplied.
    """

    @_dispatch
    def __call__(self, x):
        """Construct the mean for a design matrix.

        Args:
            x (input): Points to construct the mean at.

        Returns:
            tensor: Mean vector as a rank 2 column vector.
        """
        raise RuntimeError(f'For mean {self}, could not resolve argument "{x}".')

    @_dispatch
    def __call__(self, x: Union[Input, FDD]):
        return self(unwrap(x))

    @_dispatch
    def __call__(self, x: MultiInput):
        return B.concat(*[self(xi) for xi in x.get()], axis=0)


# Register the algebra.
@algebra.get_algebra.dispatch
def get_algebra(a: Mean):
    return Mean


class SumMean(Mean, algebra.SumFunction):
    """Sum of two means."""

    @_dispatch
    def __call__(self, x):
        return B.add(self[0](x), self[1](x))


class ProductMean(Mean, algebra.ProductFunction):
    """Product of two means."""

    @_dispatch
    def __call__(self, x):
        return B.multiply(self[0](x), self[1](x))


class ScaledMean(Mean, algebra.ScaledFunction):
    """Scaled mean."""

    @_dispatch
    def __call__(self, x):
        return B.multiply(self.scale, self[0](x))


class StretchedMean(Mean, algebra.StretchedFunction):
    """Stretched mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](B.divide(x, self.stretches[0]))


class ShiftedMean(Mean, algebra.ShiftedFunction):
    """Shifted mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](B.subtract(x, self.shifts[0]))


class SelectedMean(Mean, algebra.SelectedFunction):
    """Mean with particular input dimensions selected."""

    @_dispatch
    @uprank
    def __call__(self, x):
        return self[0](B.take(x, self.dims[0], axis=1))


class InputTransformedMean(Mean, algebra.InputTransformedFunction):
    """Input-transformed mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](uprank(self.fs[0](uprank(x))))


class OneMean(Mean, algebra.OneFunction):
    """Constant mean of `1`."""

    @_dispatch
    def __call__(self, x: B.Numeric):
        return B.ones(B.dtype(x), num_elements(x), 1)


class ZeroMean(Mean, algebra.ZeroFunction):
    """Constant mean of `0`."""

    @_dispatch
    def __call__(self, x: B.Numeric):
        return B.zeros(B.dtype(x), num_elements(x), 1)


class TensorProductMean(Mean, algebra.TensorProductFunction):
    @_dispatch
    def __call__(self, x: B.Numeric):
        return uprank(self.fs[0](uprank(x)))


class DerivativeMean(Mean, algebra.DerivativeFunction):
    """Derivative of mean."""

    @_dispatch
    @uprank
    def __call__(self, x: B.Numeric):
        import tensorflow as tf

        i = self.derivs[0]
        with tf.GradientTape() as t:
            xi = x[:, i : i + 1]
            t.watch(xi)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)
            out = B.dense(self[0](x))
            return t.gradient(out, xi, unconnected_gradients="zero")


class PosteriorMean(Mean):
    """Posterior mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes corresponding to the
            data and the input respectively.
        z (input): Locations of data.
        K_z (matrix): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    def __init__(self, m_i, m_z, k_zi, z, K_z, y):
        self.m_i = m_i
        self.m_z = m_z
        self.k_zi = k_zi
        self.z = z
        self.K_z = convert(K_z, AbstractMatrix)
        self.y = uprank(y)

    @_dispatch
    def __call__(self, x):
        diff = B.subtract(self.y, self.m_z(self.z))
        return B.add(self.m_i(x), B.iqf(self.K_z, self.k_zi(self.z, x), diff))
