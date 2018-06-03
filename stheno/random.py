# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable

from .kernel import PosteriorKernel
from .mean import ZeroMean, PosteriorMean
from .spd import SPD

__all__ = ['Normal', 'GPPrimitive', 'Random']


class Random(object):
    """A random object."""

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def rmatmul(self, other):
        raise NotImplementedError('Matrix multiplication not implemented for '
                                  '{}'.format(type(self)))

    def lmatmul(self, other):
        raise NotImplementedError('Matrix multiplication not implemented for '
                                  '{}'.format(type(self)))


class RandomProcess(Random):
    """A random process."""


class RandomVector(Random):
    """A random vector."""


class Normal(RandomVector, Referentiable):
    """Normal random variable.

    Args:
        var (matrix or instance of :class:`.spd.SPD`): Variance of the
            distribution.
        mean (column vector, optional): Mean of the distribution, defaults to
            zero.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, var, mean=None):
        self.spd = var if isinstance(var, SPD) else SPD(var)
        self.dtype = self.spd.dtype
        self.dim = B.shape(self.var)[0]
        if mean is None:
            mean = B.zeros([self.dim, 1], dtype=self.dtype)
        self.mean = mean

    @property
    def var(self):
        """Variance"""
        return self.spd.mat

    def m2(self):
        """Second moment of the distribution"""
        return self.var + B.outer(self.mean)

    def log_pdf(self, x):
        """Compute the log-pdf.

        Args:
            x (design matrix): Values to compute the log-pdf of.
        """
        if B.rank(x) != 2:
            raise ValueError('Input must have rank 2.')
        return -(self.spd.log_det() +
                 B.cast(self.dim, dtype=self.dtype) *
                 B.cast(B.log_2_pi, dtype=self.dtype) +
                 self.spd.mah_dist2(x - self.mean, sum=False)) / 2

    def entropy(self):
        """Compute the entropy."""
        return (self.spd.log_det() +
                B.cast(self.dim, dtype=self.dtype) *
                B.cast(B.log_2_pi + 1, dtype=self.dtype)) / 2

    @dispatch(Self)
    def kl(self, other):
        """Compute the KL divergence with respect to another normal
        distribution.

        Args:
            other (instance of :class:`.random.Normal`): Other normal.
        """
        return (self.spd.ratio(other.spd) +
                other.spd.mah_dist2(other.mean, self.mean) -
                B.cast(self.dim, dtype=self.dtype) +
                other.spd.log_det() - self.spd.log_det()) / 2

    @dispatch(Self)
    def w2(self, other):
        """Compute the 2-Wasserstein distance with respect to another normal
        distribution.

        Args:
            other (instance of :class:`.random.Normal`): Other normal.
        """
        root = SPD(B.dot(B.dot(self.spd.root(), other.var),
                         self.spd.root())).root()
        var_part = B.trace(self.var) + B.trace(other.var) - 2 * B.trace(root)
        mean_part = B.sum((self.mean - other.mean) ** 2)
        # The sum of `mean_part` and `var_par` should be positive, but this
        # may not be the case due to numerical errors.
        return B.abs(mean_part + var_part) ** .5

    def sample(self, num=1, noise=None):
        """Sample from the distribution.

        Args:
            num (int): Number of samples.
            noise (positive float, optional): Variance of noise to add to the
                samples.
        """
        L = self.spd.cholesky()
        e = B.randn((self.dim, num), dtype=self.dtype)
        out = B.dot(L, e) + self.mean
        if noise is not None and noise > 0:
            out += noise ** .5 * B.randn((self.dim, num), dtype=self.dtype)
        return out

    @dispatch(object)
    def __add__(self, other):
        return Normal(self.spd, self.mean + other)

    @dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(Self)
    def __add__(self, other):
        return Normal(self.spd + other.spd, self.mean + other.mean)

    @dispatch(object)
    def __mul__(self, other):
        return Normal(self.spd * other ** 2, self.mean * other)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    def rmatmul(self, other):
        return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                      B.dot(self.mean, other))

    def lmatmul(self, other):
        return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                      B.dot(other, self.mean))


class GPPrimitive(RandomProcess, Referentiable):
    """Gaussian process.

    Args:
        kernel (instance of :class:`.kernel.Kernel`): Kernel of the
            process.
        mean (instance of :class:`.mean.Mean`, optional): Mean function of the
            process. Defaults to zero.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, kernel, mean=None):
        self.mean = mean if mean else ZeroMean()
        self.kernel = kernel

    def __call__(self, x):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
           x (design matrix): Points to construct the distribution at.

        Returns:
            Instance of :class:`gptools.normal.Normal`.
        """
        return Normal(B.reg(self.kernel(x)), self.mean(x))

    def condition(self, x, y):
        """Condition the GP on a number of points.

        Args:
            x (design matrix): Locations of the points to condition on.
            y (design matrix): Values of the points to condition on.

        Returns:
            Instance of :class:`.random.GP`.
        """
        K = SPD(self.kernel(x))
        return GPPrimitive(PosteriorKernel(self, x, K),
                           PosteriorMean(self, x, K, y))

    def predict(self, x):
        """Predict at specified locations.

        Args:
            x (design matrix): Locations of the points to predict for.

        Returns:
            A tuple containing the predictive means and lower and upper 95%
            central credible interval bounds.
        """
        dist = self(x)
        mean, std = B.squeeze(dist.mean), dist.spd.diag ** .5
        return mean, mean - 2 * std, mean + 2 * std

    @dispatch(object)
    def __add__(self, other):
        return GPPrimitive(self.kernel, self.mean + other)

    @dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a GP and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(Self)
    def __add__(self, other):
        return GPPrimitive(self.kernel + other.kernel, self.mean + other.mean)

    @dispatch(object)
    def __mul__(self, other):
        return GPPrimitive(other ** 2 * self.kernel, other * self.mean)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a GP and a {}.'
                                  ''.format(type(other).__name__))

    @property
    def stationary(self):
        return self.kernel.stationary

    @property
    def var(self):
        return self.kernel.var

    @property
    def length_scale(self):
        return self.kernel.length_scale

    @property
    def period(self):
        return self.kernel.period
