from types import FunctionType

from lab import B
from matrix import AbstractMatrix, Zero
from plum import Dispatcher, convert, Union

from .util import uprank

__all__ = ["Normal"]

_dispatch = Dispatcher()


class Random:
    """A random object."""

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __div__(self, other):
        return self * (1 / other)

    def __truediv__(self, other):
        return Random.__div__(self, other)


class RandomProcess(Random):
    """A random process."""


class RandomVector(Random):
    """A random vector."""


class Normal(RandomVector):
    """Normal random variable.

    Args:
        mean (column vector, optional): Mean of the distribution. Defaults to zero.
        var (matrix): Variance of the distribution.
    """

    @_dispatch
    def __init__(
        self,
        mean: Union[B.Numeric, AbstractMatrix],
        var: Union[B.Numeric, AbstractMatrix],
    ):
        self._mean = mean
        self._mean_is_zero = None
        self._var = var

    @_dispatch
    def __init__(self, var: Union[B.Numeric, AbstractMatrix]):
        Normal.__init__(self, 0, var)

    @_dispatch
    def __init__(self, construct_mean: FunctionType, construct_var: FunctionType):
        self._mean = None
        self._construct_mean = construct_mean
        self._mean_is_zero = None
        self._var = None
        self._construct_var = construct_var

    @_dispatch
    def __init__(self, construct_var: FunctionType):
        Normal.__init__(self, lambda: 0, construct_var)

    def _resolve_mean(self, construct_zeros):
        if self._mean is None:
            self._mean = self._construct_mean()
        if self._mean_is_zero is None:
            self._mean_is_zero = self._mean is 0 or isinstance(self._mean, Zero)
        if self._mean is 0 and construct_zeros:
            self._mean = B.zeros(self.dtype, self.dim, 1)

    def _resolve_var(self):
        if self._var is None:
            self._var = self._construct_var()
        # Ensure that the variance is a structured matrix for efficient operations.
        self._var = convert(self._var, AbstractMatrix)

    @property
    def mean(self):
        """Mean."""
        self._resolve_mean(construct_zeros=True)
        return self._mean

    @property
    def mean_is_zero(self):
        """The mean is zero."""
        self._resolve_mean(construct_zeros=False)
        return self._mean_is_zero

    @property
    def var(self):
        """Variance."""
        self._resolve_var()
        return self._var

    @property
    def dtype(self):
        """Data type."""
        return B.dtype(self.var)

    @property
    def dim(self):
        """Dimensionality."""
        return B.shape(self.var)[0]

    @property
    def m2(self):
        """Second moment."""
        return self.var + B.outer(B.squeeze(self.mean))

    def marginals(self):
        """Get the marginals.

        Returns:
            tuple: A tuple containing the predictive means and lower and
                upper 95% central credible interval bounds.
        """
        mean = B.squeeze(B.dense(self.mean))
        # It can happen that the variances are slightly negative due to numerical noise.
        # Prevent NaNs from the following square root by taking the maximum with zero.
        error = 1.96 * B.sqrt(B.maximum(B.diag(self.var), B.cast(self.dtype, 0)))
        return mean, mean - error, mean + error

    def logpdf(self, x):
        """Compute the log-pdf.

        Args:
            x (input): Values to compute the log-pdf of.

        Returns:
            list[tensor]: Log-pdf for every input in `x`. If it can be
                determined that the list contains only a single log-pdf,
                then the list is flattened to a scalar.
        """
        logpdfs = (
            -(
                B.logdet(self.var)
                + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
                + B.iqf_diag(self.var, B.subtract(uprank(x), self.mean))
            )
            / 2
        )
        return logpdfs[0] if B.shape(logpdfs) == (1,) else logpdfs

    def entropy(self):
        """Compute the entropy.

        Returns:
            scalar: The entropy.
        """
        return (
            B.logdet(self.var)
            + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi + 1)
        ) / 2

    @_dispatch
    def kl(self, other: "Normal"):
        """Compute the KL divergence with respect to another normal
        distribution.

        Args:
            other (:class:`.random.Normal`): Other normal.

        Returns:
            scalar: KL divergence.
        """
        return (
            B.ratio(self.var, other.var)
            + B.iqf_diag(other.var, other.mean - self.mean)[0]
            - B.cast(self.dtype, self.dim)
            + B.logdet(other.var)
            - B.logdet(self.var)
        ) / 2

    @_dispatch
    def w2(self, other: "Normal"):
        """Compute the 2-Wasserstein distance with respect to another normal
        distribution.

        Args:
            other (:class:`.random.Normal`): Other normal.

        Returns:
            scalar: 2-Wasserstein distance.
        """
        var_root = B.root(self.var)
        root = B.root(B.matmul(var_root, other.var, var_root))
        var_part = B.trace(self.var) + B.trace(other.var) - 2 * B.trace(root)
        mean_part = B.sum((self.mean - other.mean) ** 2)
        # The sum of `mean_part` and `var_par` should be positive, but this
        # may not be the case due to numerical errors.
        return B.sqrt(B.maximum(mean_part + var_part, B.cast(self.dtype, 0)))

    def sample(self, num=1, noise=None):
        """Sample from the distribution.

        Args:
            num (int): Number of samples.
            noise (scalar, optional): Variance of noise to add to the
                samples. Must be positive.

        Returns:
            tensor: Samples as rank 2 column vectors.
        """
        var = self.var

        # Add noise.
        if noise is not None:
            var = B.add(var, B.fill_diag(noise, self.dim))

        # Perform sampling operation.
        sample = B.sample(var, num=num)
        if not self.mean_is_zero:
            sample = B.add(sample, self.mean)

        return B.dense(sample)

    @_dispatch
    def __add__(self, other: B.Numeric):
        return Normal(self.mean + other, self.var)

    @_dispatch
    def __add__(self, other: "Normal"):
        return Normal(B.add(self.mean, other.mean), B.add(self.var, other.var))

    @_dispatch
    def __mul__(self, other: B.Numeric):
        return Normal(B.multiply(self.mean, other), B.multiply(self.var, other ** 2))

    def lmatmul(self, other):
        return Normal(
            B.matmul(other, self.mean),
            B.matmul(B.matmul(other, self.var), other, tr_b=True),
        )

    def rmatmul(self, other):
        return Normal(
            B.matmul(other, self.mean, tr_a=True),
            B.matmul(B.matmul(other, self.var, tr_a=True), other),
        )
