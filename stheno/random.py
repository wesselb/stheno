from types import FunctionType

from lab import B
from matrix import AbstractMatrix, Zero, Diagonal
from plum import convert, Union
from wbml.util import indented_kv

from . import _dispatch

__all__ = ["Random", "RandomProcess", "RandomVector", "Normal"]


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
        self._var_diag = None
        self._construct_var_diag = None

    @_dispatch
    def __init__(self, var: Union[B.Numeric, AbstractMatrix]):
        Normal.__init__(self, 0, var)

    @_dispatch
    def __init__(
        self,
        mean: FunctionType,
        var: FunctionType,
        *,
        var_diag: Union[FunctionType, None] = None,
        mean_var: Union[FunctionType, None] = None,
        mean_var_diag: Union[FunctionType, None] = None,
    ):
        self._mean = None
        self._construct_mean = mean
        self._mean_is_zero = None
        self._var = None
        self._construct_var = var
        self._var_diag = None
        self._construct_var_diag = var_diag
        self._construct_mean_var = mean_var
        self._construct_mean_var_diag = mean_var_diag

    @_dispatch
    def __init__(self, var: FunctionType, **kw_args):
        Normal.__init__(self, lambda: 0, var, **kw_args)

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

    def _resolve_var_diag(self):
        if self._var_diag is None:
            if self._construct_var_diag is not None:
                self._var_diag = self._construct_var_diag()
            else:
                self._var_diag = B.diag(self.var)

    def __str__(self):
        return (
            f"<Normal:\n"
            + indented_kv(
                "mean",
                "unresolved" if self._mean is None else str(self._mean),
                suffix=",\n",
            )
            + indented_kv(
                "var",
                "unresolved" if self._var is None else str(self._var),
                suffix=">",
            )
        )

    def __repr__(self):
        return (
            f"<Normal:\n"
            + indented_kv(
                "mean",
                "unresolved" if self._mean is None else repr(self._mean),
                suffix=",\n",
            )
            + indented_kv(
                "var",
                "unresolved" if self._var is None else repr(self._var),
                suffix=">",
            )
        )

    @property
    def mean(self):
        """column vector: Mean."""
        self._resolve_mean(construct_zeros=True)
        return self._mean

    @property
    def mean_is_zero(self):
        """bool: The mean is zero."""
        self._resolve_mean(construct_zeros=False)
        return self._mean_is_zero

    @property
    def var(self):
        """matrix: Variance."""
        self._resolve_var()
        return self._var

    @property
    def var_diag(self):
        """vector: Diagonal of the variance."""
        self._resolve_var_diag()
        return self._var_diag

    @property
    def mean_var(self):
        """tuple[column vector, matrix]: Mean and variance."""
        if self._mean is not None and self._var is not None:
            return self._mean, self._var
        elif self._mean is not None:
            return self._mean, self.var
        elif self._var is not None:
            return self.mean, self._var
        else:
            if self._construct_mean_var is not None:
                self._mean, self._var = self._construct_mean_var()
                self._resolve_mean(construct_zeros=True)
                self._resolve_var()
            return self.mean, self.var

    @property
    def dtype(self):
        """dtype: Data type of the variance."""
        return B.dtype(self.var)

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.var)[0]

    @property
    def m2(self):
        """matrix: Second moment."""
        return self.var + B.outer(B.squeeze(self.mean))

    def marginals(self):
        """Get the marginals.

        Returns:
            tuple: A tuple containing the marginal means and marginal variances.
        """
        if self._mean is not None and self._var_diag is not None:
            mean, var_diag = self._mean, self._var_diag
        elif self._mean is not None:
            mean, var_diag = self._mean, self.var_diag
        elif self._var_diag is not None:
            mean, var_diag = self.mean, self._var_diag
        else:
            if self._construct_mean_var_diag is not None:
                self._mean, self._var_diag = self._construct_mean_var_diag()
                self._resolve_mean(construct_zeros=True)
            mean, var_diag = self.mean, self.var_diag
        # It can happen that the variances are slightly negative due to numerical noise.
        # Prevent NaNs from the following square root by taking the maximum with zero.
        # Also strip away any matrix structure.
        return (
            B.squeeze(B.dense(mean), axis=-1),
            B.maximum(B.dense(var_diag), B.zero(var_diag)),
        )

    def marginal_credible_bounds(self):
        """Get the marginal credible region bounds.

        Returns:
            tuple: A tuple containing the marginal means and marginal lower and
                upper 95% central credible interval bounds.
        """
        mean, var = self.marginals()
        error = 1.96 * B.sqrt(var)
        return mean, mean - error, mean + error

    def diagonalise(self):
        """Diagonalise the normal distribution by setting the correlations to zero.

        Returns:
            :class:`.Normal`: Diagonal version of the distribution.
        """
        return Normal(self.mean, Diagonal(self.var_diag))

    def logpdf(self, x):
        """Compute the log-pdf.

        Args:
            x (input): Values to compute the log-pdf of.

        Returns:
            list[tensor]: Log-pdf for every input in `x`. If it can be
                determined that the list contains only a single log-pdf,
                then the list is flattened to a scalar.
        """
        x = B.uprank(x)

        # Handle missing data. We don't handle missing data for batched computation.
        if B.rank(x) == 2 and B.shape(x, 1) == 1:
            available = B.jit_to_numpy(~B.isnan(x[:, 0]))
            if not B.all(available):
                # Take the elements of the mean, variance, and inputs corresponding to
                # the available data.
                available_mean = B.take(self.mean, available)
                available_var = B.submatrix(self.var, available)
                available_x = B.take(x, available)
                return Normal(available_mean, available_var).logpdf(available_x)

        logpdfs = (
            -(
                B.logdet(self.var)[..., None]  # Correctly line up with `iqf_diag`.
                + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
                + B.iqf_diag(self.var, B.subtract(x, self.mean))
            )
            / 2
        )
        return logpdfs[..., 0] if B.shape(logpdfs, -1) == 1 else logpdfs

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
            B.iqf_diag(other.var, other.mean - self.mean)[..., 0]
            + B.ratio(self.var, other.var)
            + B.logdet(other.var)
            - B.logdet(self.var)
            - B.cast(self.dtype, self.dim)
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

    @_dispatch
    def sample(self, state: B.RandomState, num: B.Int = 1, noise=None):
        """Sample from the distribution.

        Args:
            state (random state, optional): Random state.
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
        state, sample = B.sample(state, var, num=num)
        if not self.mean_is_zero:
            sample = B.add(sample, self.mean)

        return state, B.dense(sample)

    @_dispatch
    def sample(self, num: B.Int = 1, noise=None):
        state, sample = self.sample(
            B.global_random_state(self.dtype), num=num, noise=noise
        )
        B.set_global_random_state(state)
        return sample

    @_dispatch
    def __add__(self, other: B.Numeric):
        return Normal(B.add(self.mean, other), self.var)

    @_dispatch
    def __add__(self, other: "Normal"):
        return Normal(
            B.add(self.mean, other.mean),
            B.add(self.var, other.var),
        )

    @_dispatch
    def __mul__(self, other: B.Numeric):
        return Normal(
            B.multiply(self.mean, other),
            B.multiply(self.var, B.multiply(other, other)),
        )

    def lmatmul(self, other):
        return Normal(
            B.matmul(other, self.mean),
            B.matmul(other, self.var, other, tr_c=True),
        )

    def rmatmul(self, other):
        return Normal(
            B.matmul(other, self.mean, tr_a=True),
            B.matmul(other, self.var, other, tr_a=True),
        )


@B.dispatch
def dtype(dist: Normal):
    return B.dtype(dist.mean, dist.var)


@B.dispatch
def cast(dtype: B.DType, dist: Normal):
    return Normal(B.cast(dtype, dist.mean), B.cast(dtype, dist.var))
