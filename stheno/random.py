from lab import B
from matrix import AbstractMatrix
from plum import Self, Referentiable, PromisedType, Dispatcher, convert

from .util import uprank

__all__ = ["Normal"]


class Random(metaclass=Referentiable):
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


PromisedGP = PromisedType()


class Normal(RandomVector):
    """Normal random variable.

    Args:
        mean (tensor, optional): Mean of the distribution. Defaults to zero.
        var (matrix): Variance of the distribution.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({B.Numeric, AbstractMatrix}, {B.Numeric, AbstractMatrix})
    def __init__(self, mean, var):
        # Resolve mean and check whether it is zero.
        if mean is 0:
            # Set to an actual zero that indicates that it is all zeros. This is an
            # optimised common case.
            self._mean = 0
            self._mean_is_zero = True
        else:
            # Not useful to retain structure here.
            self._mean = B.dense(mean)
            self._mean_is_zero = False

        # Ensure that the variance is an instance of `AbstractMatrix`.
        self._var = convert(var, AbstractMatrix)

    @_dispatch({B.Numeric, AbstractMatrix})
    def __init__(self, var):
        self.__init__(0, var)

    @property
    def mean(self):
        """Mean."""
        if self._mean is 0:
            # Mean is all zeros. We now need to construct it.
            self._mean = B.zeros(self.dtype, self.dim, 1)
        return self._mean

    @property
    def var(self):
        """Variance."""
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
        mean = B.squeeze(self.mean)
        error = 1.96 * B.sqrt(B.diag(self.var))
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
                + B.iqf_diag(self.var, uprank(x) - self.mean)
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

    @_dispatch(Self)
    def kl(self, other):
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

    @_dispatch(Self)
    def w2(self, other):
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
        return B.abs(mean_part + var_part) ** 0.5

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
            # Put diagonal matrix first in the sum to ease dispatch.
            var = B.fill_diag(noise, self.dim) + self.var

        # Perform sampling operation.
        sample = B.sample(var, num=num)
        if not self._mean_is_zero:
            sample = sample + self.mean

        return B.dense(sample)

    @_dispatch(object)
    def __add__(self, other):
        return Normal(self.mean + other, self.var)

    @_dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError(
            f'Cannot add a normal and a "{type(other).__name__}".'
        )

    @_dispatch(Self)
    def __add__(self, other):
        return Normal(self.mean + other.mean, self.var + other.var)

    @_dispatch(object)
    def __mul__(self, other):
        return Normal(self.mean * other, self.var * other ** 2)

    @_dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError(
            f'Cannot multiply a normal and a f"{type(other).__name__}".'
        )

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
