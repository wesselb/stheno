# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from types import FunctionType

from lab import B
from plum import Self, Referentiable, PromisedType, type_parameter

from .cache import Cache, uprank
from .kernel import PosteriorKernel, OneKernel
from .matrix import matrix, Dispatcher, UniformlyDiagonal, Diagonal, dense, \
    Dense
from .mean import ZeroMean, PosteriorMean, OneMean
from .input import Input, At

__all__ = ['Normal', 'GPPrimitive', 'Normal1D']


class Random(object):
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


PromisedGPPrimitive = PromisedType()


class Normal(RandomVector, At, Referentiable):
    """Normal random variable.

    A normal random variable also acts as in instance of `At`, which can be
    used to specify a process at particular points.

    Args:
        var (tensor or :class:`.matrix.Dense`): Variance of the
            distribution.
        mean (tensor, optional): Mean of the distribution. Defaults to
            zero.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({B.Numeric, Dense}, [{B.Numeric, Dense, type(None)}])
    def __init__(self, var, mean=None):
        # Ensure that the variance is stored as a structured matrix.
        self._var = matrix(var)

        # Resolve mean.
        if mean is None:
            self._mean = B.zeros([self.dim, 1], dtype=self.dtype)
        else:
            self._mean = dense(mean)  # Not useful to retain structure here.

        # Set `p`, `x`, and `cache` to `None`.
        self.p = None
        self.x = None
        self.cache = None

    @_dispatch(PromisedGPPrimitive, {B.Numeric, Input}, [{Cache, type(None)}])
    def __init__(self, p, x, cache=None):
        # Set variance and mean to `None` to signify that they have to be
        # acquired from `p.kernel(x)` and `p.mean(x)`.
        self._var = None
        self._mean = None

        # Save process, the point at which it was evaluated, and the provided
        # cache.
        self.p = p
        self.x = x
        self.cache = Cache() if cache is None else cache

    def get(self):
        """Get the point at which the the process was evaluated to construct
        the distribution.

        This method must be implemented to act as an instance of `Act`.

        Returns:
            input: Point at which the process was evaluated.
        """
        if self.p is None:
            raise RuntimeError('Normal distribution is not a '
                               'finite-dimensional distribution from a '
                               'process.')
        return self.x

    def __new__(cls, *args, **kw_args):
        # Since we have subtyped a parametric type, during instantiating the
        # parametric type will attempt to generate a new parametric type.
        # Prevent this. Simply redirect the call to `Normal.__new__` to that of
        # its base: `RandomVector`.
        return RandomVector.__new__(cls)

    @property
    def var(self):
        """Variance."""
        if self._var is None:
            # Variance must be acquired from the saved process.
            self._var = self.p.kernel(self.x, self.cache)
        return self._var

    @property
    def mean(self):
        """Mean."""
        if self._mean is None:
            # Mean must be acquired from the saved process. It is not useful
            # to retain structure.
            self._mean = dense(self.p.mean(self.x, self.cache))
        return self._mean

    @property
    def dtype(self):
        """Data type."""
        return B.dtype(self.var)

    @property
    def dim(self):
        """Dimensionality."""
        return B.shape(self.var)[0]

    def m2(self):
        """Second moment."""
        return self.var + B.outer(self.mean)

    def logpdf(self, x):
        """Compute the log-pdf.

        Args:
            x (input): Values to compute the log-pdf of.
            
        Returns:
            list[tensor]: Log-pdf for every input in `x`.
        """
        return -(B.logdet(self.var) +
                 B.cast(self.dim, dtype=self.dtype) *
                 B.cast(B.log_2_pi, dtype=self.dtype) +
                 B.qf_diag(self.var, uprank(x) - self.mean)) / 2

    def entropy(self):
        """Compute the entropy.
        
        Returns:
            scalar: The entropy.
        """
        return (B.logdet(self.var) +
                B.cast(self.dim, dtype=self.dtype) *
                B.cast(B.log_2_pi + 1, dtype=self.dtype)) / 2

    @_dispatch(Self)
    def kl(self, other):
        """Compute the KL divergence with respect to another normal
        distribution.

        Args:
            other (:class:`.random.Normal`): Other normal.

        Returns:
            scalar: KL divergence.
        """
        return (B.ratio(self.var, other.var) +
                B.qf_diag(other.var, other.mean - self.mean)[0] -
                B.cast(self.dim, dtype=self.dtype) +
                B.logdet(other.var) - B.logdet(self.var)) / 2

    @_dispatch(Self)
    def w2(self, other):
        """Compute the 2-Wasserstein distance with respect to another normal
        distribution.

        Args:
            other (:class:`.random.Normal`): Other normal.

        Returns:
            scalar: 2-Wasserstein distance.
        """
        root = B.root(B.matmul(B.root(self.var), other.var, B.root(self.var)))
        var_part = B.trace(self.var) + B.trace(other.var) - 2 * B.trace(root)
        mean_part = B.sum((self.mean - other.mean) ** 2)
        # The sum of `mean_part` and `var_par` should be positive, but this
        # may not be the case due to numerical errors.
        return B.abs(mean_part + var_part) ** .5

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

        # Add noise if requested for.
        if noise is not None:
            var += UniformlyDiagonal.from_(noise, self.var)

        # Perform sampling operation.
        return B.sample(var, num) + self.mean

    @_dispatch(object)
    def __add__(self, other):
        return Normal(self.var, self.mean + other)

    @_dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @_dispatch(Self)
    def __add__(self, other):
        return Normal(self.var + other.var, self.mean + other.mean)

    @_dispatch(object)
    def __mul__(self, other):
        return Normal(self.var * other ** 2, self.mean * other)

    @_dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    def rmatmul(self, other):
        return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                      B.dot(self.mean, other))

    def lmatmul(self, other):
        return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                      B.dot(other, self.mean))

    def __repr__(self):
        if self.p is None:
            return RandomVector.__repr__(self)
        else:
            return '{!r}({!r})'.format(self.p, self.x)

    def __str__(self):
        if self.p is None:
            return RandomVector.__str__(self)
        else:
            return '{!s}({!s})'.format(self.p, self.x)


# Make sure the type parameter can be extracted from a `Normal`.
@type_parameter.extend(Normal)
def type_parameter(a):
    if a.p is None:
        raise RuntimeError('Normal distribution is not a finite-dimensional '
                           'distribution from a process.')
    return a.p


class Normal1D(Normal, Referentiable):
    """A one-dimensional version of :class:`.random.Normal` with convenient
    broadcasting behaviour.

    Args:
        var (scalar or vector): Variance of the distribution.
        mean (scalar or vector): Mean of the distribution, defaults to
            zero.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, var, mean=None):
        var = B.array(var)

        # Consider all various ranks of `var` and `mean` for convenient
        # broadcasting behaviour.
        if mean is not None:
            mean = B.array(mean)

            if B.rank(var) == 1:
                if B.rank(mean) == 0:
                    mean = mean * \
                           B.ones([B.shape(var)[0], 1], dtype=B.dtype(var))
                elif B.rank(mean) == 1:
                    mean = mean[:, None]
                else:
                    raise ValueError('Invalid rank {} of mean.'
                                     ''.format(B.rank(mean)))
                var = Diagonal(var)

            elif B.rank(var) == 0:
                if B.rank(mean) == 0:
                    mean = mean * B.ones([1, 1], dtype=B.dtype(var))
                    var = UniformlyDiagonal(var, 1)
                elif B.rank(mean) == 1:
                    mean = mean[:, None]
                    var = UniformlyDiagonal(var, B.shape(mean)[0])
                else:
                    raise ValueError('Invalid rank {} of mean.'
                                     ''.format(B.rank(mean)))

            else:
                raise ValueError('Invalid rank {} of variance.'
                                 ''.format(B.rank(var)))
        else:
            if B.rank(var) == 0:
                var = UniformlyDiagonal(var, 1)
            elif B.rank(var) == 1:
                var = Diagonal(var)
            else:
                raise ValueError('Invalid rank {} of variance.'
                                 ''.format(B.rank(var)))

        Normal.__init__(self, var, mean)


class GPPrimitive(RandomProcess, Referentiable):
    """Gaussian process.

    Args:
        kernel (:class:`.kernel.Kernel`): Kernel of the
            process.
        mean (:class:`.mean.Mean`, optional): Mean function of the
            process. Defaults to zero.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, kernel, mean=None):
        self._kernel = kernel
        self._mean = mean

    @property
    def kernel(self):
        """Kernel of the GP."""
        # Resolve kernel lazily.
        if isinstance(self._kernel, (B.Numeric, FunctionType)):
            self._kernel = self._kernel * OneKernel()
        return self._kernel

    @property
    def mean(self):
        """Mean function of the GP."""
        # Resolve mean lazily.
        if self._mean is None:
            self._mean = ZeroMean()
        elif isinstance(self._mean, (B.Numeric, FunctionType)):
            self._mean = self._mean * OneMean()
        return self._mean

    def __call__(self, x, cache=None):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
            x (input): Points to construct the distribution at.
            cache (:class:`.cache.Cache`, optional): Cache.

        Returns:
            :class:`.random.Normal`: Finite-dimensional distribution.
        """
        cache = Cache() if cache is None else cache
        return Normal(self, x, cache)

    @_dispatch(object, object)
    def condition(self, x, y):
        """Condition the GP on a number of points.

        Args:
            x (input): Locations of the points to condition on.
            y (tensor): Values of the points to condition on.

        Returns:
            :class:`.random.GP`: Conditioned GP.
        """
        K = matrix(self.kernel(x))
        return GPPrimitive(PosteriorKernel(self.kernel,
                                           self.kernel,
                                           self.kernel, x, K),
                           PosteriorMean(self.mean,
                                         self.mean,
                                         self.kernel, x, K, y))

    def predict(self, x, cache=None):
        """Predict at specified locations.

        Args:
            x (design matrix): Locations of the points to predict for.
            cache (:class:`.cache.Cache`, optional): Cache.

        Returns:
            tuple: A tuple containing the predictive means and lower and
            upper 95% central credible interval bounds.
        """
        cache = Cache() if cache is None else cache
        mean = B.squeeze(dense(self.mean(x, cache)))
        std = B.squeeze(dense(self.kernel.elwise(x, cache))) ** .5
        return mean, mean - 2 * std, mean + 2 * std

    @_dispatch(object)
    def __add__(self, other):
        return GPPrimitive(self.kernel, self.mean + other)

    @_dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a GP and a {}.'
                                  ''.format(type(other).__name__))

    @_dispatch(Self)
    def __add__(self, other):
        return GPPrimitive(self.kernel + other.kernel, self.mean + other.mean)

    @_dispatch(object)
    def __mul__(self, other):
        return GPPrimitive(other ** 2 * self.kernel, other * self.mean)

    @_dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a GP and a {}.'
                                  ''.format(type(other).__name__))

    @property
    def stationary(self):
        """Stationarity of the GP."""
        return self.kernel.stationary

    @property
    def var(self):
        """Variance of the GP."""
        return self.kernel.var

    @property
    def length_scale(self):
        """Length scale of the GP."""
        return self.kernel.length_scale

    @property
    def period(self):
        """Period of the GP."""
        return self.kernel.period

    def __str__(self):
        return self.display()

    def __repr__(self):
        return self.display()

    def display(self, formatter=lambda x: x):
        """Display the GP.

        Args:
            formatter (function, optional): Function to format values.

        Returns:
            str: GP as a string.
        """
        return 'GP({}, {})'.format(self.kernel.display(formatter),
                                   self.mean.display(formatter))

    def stretch(self, stretch):
        """Stretch the GP. See :meth:`.graph.Graph.stretch`."""
        return GPPrimitive(self.kernel.stretch(stretch),
                           self.mean.stretch(stretch))

    def shift(self, shift):
        """Shift the GP. See :meth:`.graph.Graph.shift`."""
        return GPPrimitive(self.kernel.shift(shift),
                           self.mean.shift(shift))

    def select(self, *dims):
        """Select dimensions from the input. See :meth:`.graph.Graph.select`."""
        return GPPrimitive(self.kernel.select(dims),
                           self.mean.select(dims))

    def transform(self, f):
        """Input transform the GP. See :meth:`.graph.Graph.transform`."""
        return GPPrimitive(self.kernel.transform(f),
                           self.mean.transform(f))

    def diff(self, deriv=0):
        """Differentiate the GP. See :meth:`.graph.Graph.diff`."""
        return GPPrimitive(self.kernel.diff(deriv),
                           self.mean.diff(deriv))


PromisedGPPrimitive.deliver(GPPrimitive)
