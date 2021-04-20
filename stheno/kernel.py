import logging

import algebra
import numpy as np
from algebra.util import identical, to_tensor
from lab import B
from matrix import Dense, LowRank, Diagonal, Zero, Constant, AbstractMatrix
from plum import Dispatcher, convert, Union

from . import PromisedFDD as FDD
from .input import Input, Unique, WeightedUnique, MultiInput
from .util import num_elements, uprank

__all__ = [
    "Kernel",
    "OneKernel",
    "ZeroKernel",
    "ScaledKernel",
    "EQ",
    "RQ",
    "Matern12",
    "Exp",
    "Matern32",
    "Matern52",
    "Delta",
    "FixedDelta",
    "Linear",
    "DerivativeKernel",
    "DecayingKernel",
    "LogKernel",
]

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch
def expand(xs: Union[tuple, list]):
    """Expand a sequence to the same element repeated twice if there is only one
    element.

    Args:
        xs (tuple or list): Sequence to expand.

    Returns:
        tuple or list: `xs * 2` or `xs`.
    """
    return xs * 2 if len(xs) == 1 else xs


@_dispatch
def unwrap(x: Input):
    """Unwrap a wrapped input.

    Args:
        x (input): Wrapped input.

    Returns:
        tensor: Unwrapped input.
    """
    return x.get()


@_dispatch
def unwrap(fdd: FDD):
    return fdd.x


class Kernel(algebra.Function):
    """Kernel function.

    Kernels can be added and multiplied.
    """

    @_dispatch
    def __call__(self, x, y):
        """Construct the kernel matrix between all `x` and `y`.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first
                argument.

        Returns:
            matrix: Kernel matrix.
        """
        raise RuntimeError(
            f'For kernel "{self}", could not resolve arguments "{x}" and "{y}".'
        )

    @_dispatch
    def __call__(self, x):
        return self(x, x)

    @_dispatch
    def __call__(self, x: Union[Input, FDD], y: Union[Input, FDD]):
        return self(unwrap(x), unwrap(y))

    @_dispatch
    def __call__(self, x: Union[Input, FDD], y):
        return self(unwrap(x), y)

    @_dispatch
    def __call__(self, x, y: Union[Input, FDD]):
        return self(x, unwrap(y))

    @_dispatch(precedence=1)
    def __call__(self, x: MultiInput, y):
        return self(x, MultiInput(y))

    @_dispatch(precedence=1)
    def __call__(self, x, y: MultiInput):
        return self(MultiInput(x), y)

    @_dispatch
    def __call__(self, x: MultiInput, y: MultiInput):
        return B.block(*[[self(xi, yi) for yi in y.get()] for xi in x.get()])

    @_dispatch
    def elwise(self, x, y):
        """Construct the kernel vector `x` and `y` element-wise.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first
                argument.

        Returns:
            tensor: Kernel vector as a rank 2 column vector.
        """
        # TODO: Throw warning.
        return B.expand_dims(B.diag(self(x, y)), axis=1)

    @_dispatch
    def elwise(self, x):
        return self.elwise(x, x)

    @_dispatch
    def elwise(self, x: Union[Input, FDD], y: Union[Input, FDD]):
        return self.elwise(unwrap(x), unwrap(y))

    @_dispatch
    def elwise(self, x: Union[Input, FDD], y):
        return self.elwise(unwrap(x), y)

    @_dispatch
    def elwise(self, x, y: Union[Input, FDD]):
        return self.elwise(x, unwrap(y))

    @_dispatch(precedence=1)
    def elwise(self, x: MultiInput, y):
        raise ValueError("Unclear combination of arguments given to `Kernel.elwise`.")

    @_dispatch(precedence=1)
    def elwise(self, x, y: MultiInput):
        raise ValueError("Unclear combination of arguments given to `Kernel.elwise`.")

    @_dispatch
    def elwise(self, x: MultiInput, y: MultiInput):
        if len(x.get()) != len(y.get()):
            raise ValueError(
                "Kernel.elwise must be called with similarly sized MultiInputs."
            )
        return B.concat(
            *[self.elwise(xi, yi) for xi, yi in zip(x.get(), y.get())], axis=0
        )

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor, optional): Period. Defaults to `1`.

        Returns:
            :class:`.kernel.Kernel`: Periodic version of the kernel.
        """
        return periodicise(self, period)

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


# Register the algebra.
@algebra.get_algebra.dispatch
def get_algebra(a: Kernel):
    return Kernel


class OneKernel(Kernel, algebra.OneFunction):
    """Constant kernel of `1`."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Constant(B.one(x), num_elements(x), num_elements(y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return B.ones(B.dtype(x), num_elements(x), 1)

    @property
    def _stationary(self):
        return True


class ZeroKernel(Kernel, algebra.ZeroFunction):
    """Constant kernel of `0`."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return B.zeros(B.dtype(x), num_elements(x), 1)

    @property
    def _stationary(self):
        return True


class ScaledKernel(Kernel, algebra.ScaledFunction):
    """Scaled kernel."""

    @_dispatch
    def __call__(self, x, y):
        return self._compute(self[0](x, y))

    @_dispatch
    def elwise(self, x, y):
        return self._compute(self[0].elwise(x, y))

    def _compute(self, K):
        return B.multiply(self.scale, K)

    @property
    def _stationary(self):
        return self[0].stationary


class SumKernel(Kernel, algebra.SumFunction):
    """Sum of kernels."""

    @_dispatch
    def __call__(self, x, y):
        return B.add(self[0](x, y), self[1](x, y))

    @_dispatch
    def elwise(self, x, y):
        return B.add(self[0].elwise(x, y), self[1].elwise(x, y))

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary


class ProductKernel(Kernel, algebra.ProductFunction):
    """Product of two kernels."""

    @_dispatch
    def __call__(self, x, y):
        return B.multiply(self[0](x, y), self[1](x, y))

    @_dispatch
    def elwise(self, x, y):
        return B.multiply(self[0].elwise(x, y), self[1].elwise(x, y))

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary


class StretchedKernel(Kernel, algebra.StretchedFunction):
    """Stretched kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return self[0](*self._compute(x, y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self[0].elwise(*self._compute(x, y))

    def _compute(self, x, y):
        stretches1, stretches2 = expand(self.stretches)
        return B.divide(x, stretches1), B.divide(y, stretches2)

    @property
    def _stationary(self):
        if len(self.stretches) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "StretchedKernel"):
        identical_stretches = identical(expand(self.stretches), expand(other.stretches))
        return self[0] == other[0] and identical_stretches


class ShiftedKernel(Kernel, algebra.ShiftedFunction):
    """Shifted kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return self[0](*self._compute(x, y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self[0].elwise(*self._compute(x, y))

    def _compute(self, x, y):
        shifts1, shifts2 = expand(self.shifts)
        return B.subtract(x, shifts1), B.subtract(y, shifts2)

    @property
    def _stationary(self):
        if len(self.shifts) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "ShiftedKernel"):
        identical_shifts = identical(expand(self.shifts), expand(other.shifts))
        return self[0] == other[0] and identical_shifts


class SelectedKernel(Kernel, algebra.SelectedFunction):
    """Kernel with particular input dimensions selected."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return self[0](*self._compute(x, y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self[0].elwise(*self._compute(x, y))

    @uprank
    def _compute(self, x, y):
        dims1, dims2 = expand(self.dims)
        x = x if dims1 is None else B.take(x, dims1, axis=1)
        y = y if dims2 is None else B.take(y, dims2, axis=1)
        return x, y

    @property
    def _stationary(self):
        if len(self.dims) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "SelectedKernel"):
        return self[0] == other[0] and identical(expand(self.dims), expand(other.dims))


class InputTransformedKernel(Kernel, algebra.InputTransformedFunction):
    """Input-transformed kernel."""

    @_dispatch
    def __call__(self, x, y):
        return self[0](*self._compute(x, y))

    @_dispatch
    def elwise(self, x, y):
        return self[0].elwise(*self._compute(x, y))

    def _compute(self, x, y):
        f1, f2 = expand(self.fs)
        x = x if f1 is None else f1(uprank(x))
        y = y if f2 is None else f2(uprank(y))
        return x, y

    @_dispatch
    def __eq__(self, other: "InputTransformedKernel"):
        return self[0] == other[0] and identical(expand(self.fs), expand(other.fs))


class PeriodicKernel(Kernel, algebra.WrappedFunction):
    """Periodic kernel.

    Args:
        k (:class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    def __init__(self, k, period):
        algebra.WrappedFunction.__init__(self, k)
        self.period = to_tensor(period)

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return self[0](*self._compute(x, y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self[0].elwise(*self._compute(x, y))

    def _compute(self, x, y):
        @uprank
        def feature_map(z):
            z = B.divide(B.multiply(B.multiply(z, 2), B.pi), self.period)
            return B.concat(B.sin(z), B.cos(z), axis=1)

        return feature_map(x), feature_map(y)

    @property
    def _stationary(self):
        return self[0].stationary

    def render_wrap(self, e, formatter):
        return f"{e} per {formatter(self.period)}"

    @_dispatch
    def __eq__(self, other: "PeriodicKernel"):
        return self[0] == other[0] and identical(self.period, other.period)


class EQ(Kernel):
    """Exponentiated quadratic kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Dense(self._compute(B.pw_dists2(x, y)))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self._compute(B.ew_dists2(x, y))

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "EQ"):
        return True


class RQ(Kernel):
    """Rational quadratic kernel.

    Args:
        alpha (scalar): Shape of the prior over length scales. Determines the
            weight of the tails of the kernel. Must be positive.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Dense(self._compute(B.pw_dists2(x, y)))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self._compute(B.ew_dists2(x, y))

    def _compute(self, dists2):
        return (1 + 0.5 * dists2 / self.alpha) ** (-self.alpha)

    def render(self, formatter):
        return f"RQ({formatter(self.alpha)})"

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "RQ"):
        return B.all(self.alpha == other.alpha)


class Exp(Kernel):
    """Exponential kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Dense(B.exp(-B.pw_dists(x, y)))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return B.exp(-B.ew_dists(x, y))

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Exp"):
        return True


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel):
    """Matern--3/2 kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Dense(self._compute(B.pw_dists(x, y)))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self._compute(B.ew_dists(x, y))

    def _compute(self, dists):
        r = 3 ** 0.5 * dists
        return (1 + r) * B.exp(-r)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Matern32"):
        return True


class Matern52(Kernel):
    """Matern--5/2 kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        return Dense(self._compute(B.pw_dists(x, y)))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return self._compute(B.ew_dists(x, y))

    def _compute(self, dists):
        r1 = 5 ** 0.5 * dists
        r2 = 5 * dists ** 2 / 3
        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Matern52"):
        return True


class Delta(Kernel):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
            Defaults to `1e-10`.
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        if x is y:
            return B.fill_diag(B.one(x), num_elements(x))
        else:
            return Dense(self._compute(B.pw_dists2(x, y)))

    @_dispatch
    def __call__(self, x: Unique, y: Unique):
        x, y = x.get(), y.get()
        if x is y:
            return B.fill_diag(B.one(x), num_elements(x))
        else:
            return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def __call__(self, x: WeightedUnique, y: WeightedUnique):
        w_x, w_y = x.w, y.w
        x, y = x.get(), y.get()
        if x is y:
            return Diagonal(1 / w_x)
        else:
            return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def __call__(self, x: Unique, y):
        x = x.get()
        return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def __call__(self, x, y: Unique):
        y = y.get()
        return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def elwise(self, x: Unique, y: Unique):
        x, y = x.get(), y.get()
        if x is y:
            return B.ones(B.dtype(x), num_elements(x), 1)
        else:
            return B.zeros(B.dtype(x), num_elements(x), 1)

    @_dispatch
    def elwise(self, x: WeightedUnique, y: WeightedUnique):
        w_x, w_y = x.w, y.w
        x, y = x.get(), y.get()
        if x is y:
            return uprank(1 / w_x)
        else:
            return B.zeros(B.dtype(x), num_elements(x), 1)

    @_dispatch
    def elwise(self, x: Unique, y):
        x = x.get()
        return B.zeros(B.dtype(x), num_elements(x), 1)

    @_dispatch
    def elwise(self, x, y: Unique):
        return B.zeros(B.dtype(x), num_elements(x), 1)

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        if x is y:
            return B.ones(B.dtype(x), num_elements(x), 1)
        else:
            return self._compute(B.ew_dists2(x, y))

    def _compute(self, dists2):
        dtype = B.dtype(dists2)
        return B.cast(dtype, B.lt(dists2, B.cast(dtype, self.epsilon)))

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Delta"):
        return self.epsilon == other.epsilon


class FixedDelta(Kernel):
    """Kronecker delta kernel that produces a diagonal matrix with given
    noises if and only if the inputs are identical and of the right shape.

    Args:
        noises (vector): Noises.
    """

    def __init__(self, noises):
        self.noises = noises

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        if x is y and num_elements(x) == B.shape(self.noises)[0]:
            return Diagonal(self.noises)
        else:
            return Zero(B.dtype(x), num_elements(x), num_elements(y))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        if x is y and num_elements(x) == B.shape(self.noises)[0]:
            return uprank(self.noises)
        else:
            return Zero(B.dtype(x), num_elements(x), 1)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "FixedDelta"):
        return B.all(self.noises == other.noises)


class Linear(Kernel):
    """Linear kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        if x is y:
            return LowRank(uprank(x))
        else:
            return LowRank(left=uprank(x), right=uprank(y))

    @_dispatch
    @uprank
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return B.expand_dims(B.sum(B.multiply(x, y), axis=1), axis=1)

    @property
    def _stationary(self):
        return False

    @_dispatch
    def __eq__(self, other: "Linear"):
        return True


class DecayingKernel(Kernel):
    """Decaying kernel.

    Args:
        alpha (tensor): Shape of the gamma distribution governing the
            distribution of decaying exponentials.
        beta (tensor): Rate of the gamma distribution governing the
            distribution of decaying exponentials.
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        pw_sums_raised = B.power(B.pw_sums(B.add(x, self.beta), y), self.alpha)
        return Dense(B.divide(self._compute_beta_raised(), pw_sums_raised))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        return B.divide(
            self._compute_beta_raised(),
            B.power(B.ew_sums(B.add(x, self.beta), y), self.alpha),
        )

    def _compute_beta_raised(self):
        beta_norm = B.sqrt(
            B.maximum(B.sum(B.power(self.beta, 2)), B.cast(B.dtype(self.beta), 1e-30))
        )
        return B.power(beta_norm, self.alpha)

    def render(self, formatter):
        return f"DecayingKernel({formatter(self.alpha)}, {formatter(self.beta)})"

    @_dispatch
    def __eq__(self, other: "DecayingKernel"):
        return B.all(self.alpha == other.alpha) and B.all(self.beta == other.beta)


class LogKernel(Kernel):
    """Logarithm kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        dists = B.maximum(B.pw_dists(x, y), 1e-10)
        return Dense(B.divide(B.log(dists + 1), dists))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        dists = B.maximum(B.ew_dists(x, y), 1e-10)
        return B.divide(B.log(dists + 1), dists)

    def render(self, formatter):
        return "LogKernel()"

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "LogKernel"):
        return True


class PosteriorKernel(Kernel):
    """Posterior kernel.

    Args:
        k_ij (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the left input and the right input respectively.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the left input respectively.
        k_zj (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the right input respectively.
        z (input): Locations of data.
        K_z (matrix): Kernel matrix of data.
    """

    def __init__(self, k_ij, k_zi, k_zj, z, K_z):
        self.k_ij = k_ij
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.K_z = convert(K_z, AbstractMatrix)

    @_dispatch
    def __call__(self, x, y):
        return B.subtract(
            self.k_ij(x, y), B.iqf(self.K_z, self.k_zi(self.z, x), self.k_zj(self.z, y))
        )

    @_dispatch
    def elwise(self, x, y):
        iqf_diag = B.iqf_diag(self.K_z, self.k_zi(self.z, x), self.k_zj(self.z, y))
        return B.subtract(self.k_ij.elwise(x, y), B.expand_dims(iqf_diag, axis=1))


class CorrectiveKernel(Kernel):
    """Kernel that adds the corrective variance in sparse conditioning.

    Args:
        k_zi (:class:`.kernel.Kernel`): Kernel between the processes corresponding to
            the left input and the inducing points respectively.
        k_zj (:class:`.kernel.Kernel`): Kernel between the processes corresponding to
            the right input and the inducing points respectively.
        z (input): Locations of the inducing points.
        A (tensor): Corrective matrix.
        L (tensor): Kernel matrix of the inducing points.
    """

    def __init__(self, k_zi, k_zj, z, A, K_z):
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.A = A
        self.L = B.cholesky(convert(K_z, AbstractMatrix))

    @_dispatch
    def __call__(self, x, y):
        return B.iqf(
            self.A,
            B.solve(self.L, self.k_zi(self.z, x)),
            B.solve(self.L, self.k_zj(self.z, y)),
        )

    @_dispatch
    def elwise(self, x, y):
        return B.iqf_diag(
            self.A,
            B.solve(self.L, self.k_zi(self.z, x)),
            B.solve(self.L, self.k_zj(self.z, y)),
        )[:, None]


def dkx(k_elwise, i):
    """Construct the derivative of a kernel with respect to its first
    argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Derivative of the kernel with respect to its first argument.
    """

    @uprank
    def _dkx(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            # Get the numbers of inputs.
            nx = num_elements(x)
            ny = num_elements(y)

            # Copy the input `ny` times to efficiently compute many derivatives.
            xis = tf.identity_n([x[:, i : i + 1]] * ny)
            t.watch(xis)

            # Tile inputs for batched computation.
            x = B.tile(x, ny, 1)
            y = B.reshape(B.tile(y, 1, nx), ny * nx, -1)

            # Insert tracked dimension, which is different for every tile.
            xi = B.concat(*xis, axis=0)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)

            # Perform the derivative computation.
            out = B.dense(k_elwise(x, y))
            grads = t.gradient(out, xis, unconnected_gradients="zero")
            return B.concat(*grads, axis=1)

    return _dkx


def dkx_elwise(k_elwise, i):
    """Construct the element-wise derivative of a kernel with respect to
    its first argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Element-wise derivative of the kernel with respect to its
            first argument.
    """

    @uprank
    def _dkx_elwise(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            xi = x[:, i : i + 1]
            t.watch(xi)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)
            out = B.dense(k_elwise(x, y))
            return t.gradient(out, xi, unconnected_gradients="zero")

    return _dkx_elwise


def dky(k_elwise, i):
    """Construct the derivative of a kernel with respect to its second
    argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Derivative of the kernel with respect to its second argument.
    """

    @uprank
    def _dky(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            # Get the numbers of inputs.
            nx = num_elements(x)
            ny = num_elements(y)

            # Copy the input `nx` times to efficiently compute many derivatives.
            yis = tf.identity_n([y[:, i : i + 1]] * nx)
            t.watch(yis)

            # Tile inputs for batched computation.
            x = B.reshape(B.tile(x, 1, ny), nx * ny, -1)
            y = B.tile(y, nx, 1)

            # Insert tracked dimension, which is different for every tile.
            yi = B.concat(*yis, axis=0)
            y = B.concat(y[:, :i], yi, y[:, i + 1 :], axis=1)

            # Perform the derivative computation.
            out = B.dense(k_elwise(x, y))
            grads = t.gradient(out, yis, unconnected_gradients="zero")
            return B.transpose(B.concat(*grads, axis=1))

    return _dky


def dky_elwise(k_elwise, i):
    """Construct the element-wise derivative of a kernel with respect to
    its second argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Element-wise derivative of the kernel with respect to its
            second argument.
    """

    @uprank
    def _dky_elwise(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            yi = y[:, i : i + 1]
            t.watch(yi)
            y = B.concat(y[:, :i], yi, y[:, i + 1 :], axis=1)
            out = B.dense(k_elwise(x, y))
            return t.gradient(out, yi, unconnected_gradients="zero")

    return _dky_elwise


def perturb(x):
    """Slightly perturb a tensor.

    Args:
        x (tensor): Tensor to perturb.

    Returns:
        tensor: `x`, but perturbed.
    """
    dtype = convert(B.dtype(x), B.NPDType)
    if dtype == np.float64:
        return 1e-20 + x * (1 + 1e-14)
    elif dtype == np.float32:
        return 1e-20 + x * (1 + 1e-7)
    else:
        raise ValueError(f"Cannot perturb a tensor of data type {B.dtype(x)}.")


class DerivativeKernel(Kernel, algebra.DerivativeFunction):
    """Derivative of kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        i, j = expand(self.derivs)
        k = self[0]

        # Prevent that `x` equals `y` to stabilise nested gradients.
        y = perturb(y)

        if i is not None and j is not None:
            # Derivative with respect to both `x` and `y`.
            return Dense(dky(dkx_elwise(k.elwise, i), j)(x, y))

        elif i is not None and j is None:
            # Derivative with respect to `x`.
            return Dense(dkx(k.elwise, i)(x, y))

        elif i is None and j is not None:
            # Derivative with respect to `y`.
            return Dense(dky(k.elwise, j)(x, y))

        else:
            raise RuntimeError("No derivative specified.")

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        i, j = expand(self.derivs)
        k = self[0]

        # Prevent that `x` equals `y` to stabilise nested gradients.
        y = perturb(y)

        if i is not None and j is not None:
            # Derivative with respect to both `x` and `y`.
            return dky_elwise(dkx_elwise(k.elwise, i), j)(x, y)

        elif i is not None and j is None:
            # Derivative with respect to `x`.
            return dkx_elwise(k.elwise, i)(x, y)

        elif i is None and j is not None:
            # Derivative with respect to `y`.
            return dky_elwise(k.elwise, j)(x, y)

        else:
            raise RuntimeError("No derivative specified.")

    @property
    def _stationary(self):
        # NOTE: In the one-dimensional case, if derivatives with respect to both
        # arguments are taken, then the result is in fact stationary.
        return False

    @_dispatch
    def __eq__(self, other: "DerivativeKernel"):
        identical_derivs = identical(expand(self.derivs), expand(other.derivs))
        return self[0] == other[0] and identical_derivs


class TensorProductKernel(Kernel, algebra.TensorProductFunction):
    """Tensor product kernel."""

    @_dispatch
    def __call__(self, x: B.Numeric, y: B.Numeric):
        f1, f2 = expand(self.fs)
        if x is y and f1 is f2:
            return LowRank(uprank(f1(uprank(x))))
        else:
            return LowRank(left=uprank(f1(uprank(x))), right=uprank(f2(uprank(y))))

    @_dispatch
    def elwise(self, x: B.Numeric, y: B.Numeric):
        f1, f2 = expand(self.fs)
        if x is y and f1 is f2:
            return B.power(uprank(f1(uprank(x))), 2)
        else:
            return B.multiply(uprank(f1(uprank(x))), uprank(f2(uprank(y))))

    @_dispatch
    def __eq__(self, other: "TensorProductKernel"):
        return identical(expand(self.fs), expand(other.fs))


class ReversedKernel(Kernel, algebra.ReversedFunction):
    """Reversed kernel.

    Evaluates with its arguments reversed.
    """

    @_dispatch
    def __call__(self, x, y):
        return B.transpose(self[0](y, x))

    @_dispatch
    def elwise(self, x, y):
        return self[0].elwise(y, x)

    @property
    def _stationary(self):
        return self[0].stationary


# Periodicise kernels.


@_dispatch
def periodicise(a: Kernel, b):
    return PeriodicKernel(a, b)


@_dispatch
def periodicise(a: ZeroKernel, b):
    return a


# Make shifting synergise with stationary kernels.


@algebra.shift.dispatch
def shift(a: Kernel, *shifts):
    if a.stationary and len(shifts) == 1:
        return a
    else:
        return ShiftedKernel(a, *shifts)
