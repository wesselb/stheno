# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable

from stheno import Dense, ZeroMean, PosteriorMean, PosteriorKernel, SPD

__all__ = ['Normal', 'GP']


class Random(object):
    """A random object."""

    def __radd__(self, other):
        return self + other


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
        self.spd = var if isinstance(var, SPD) else Dense(var)
        self.dtype = self.var.dtype
        self.dim = B.shape(self.var)[0]
        if mean is None:
            self.mean = B.zeros([self.spd.shape[0], 1], dtype=self.dtype)
        else:
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
                other.spd.mah_dist2(other.mean, self.mean) - self.dim +
                other.spd.log_det() - self.spd.log_det()) / 2

    @dispatch(Self)
    def w2(self, other):
        """Compute the 2-Wasserstein distance with respect to another normal
        distribution.

        Args:
            other (instance of :class:`.random.Normal`): Other normal.
        """
        root = Dense(B.dot(B.dot(self.spd.root(), other.var),
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

    @dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __add__(self, other):
        return Normal(self.var, self.mean + other)

    @dispatch(Self)
    def __add__(self, other):
        # TODO: implement + for SPD
        return Normal(self.var + other.var, self.mean + other.mean)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(Random)
    def __rmul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __mul__(self, other):
        if B.is_scalar(other):
            return Normal(self.var * other ** 2, self.mean * other)
        else:
            return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                          B.dot(self.mean, other))

    @dispatch(object)
    def __rmul__(self, other):
        if B.is_scalar(other):
            return Normal(self.var * other ** 2, self.mean * other)
        else:
            return Normal(B.dot(B.dot(other, self.var), other, tr_b=True),
                          B.dot(other, self.mean))


class GP(RandomProcess, Referentiable):
    """Gaussian process.

    Args:
        kernel (instance of :class:`.kernel.Kernel`): Kernel of the
            process.
        mean (column vector, optional): Mean function of the process. Defaults
            to zero.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, kernel, mean=None):
        self.mean = mean if mean else ZeroMean()
        self.kernel = kernel

    def __call__(self, x, noise=None):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
           x (design matrix): Points to construct the distribution at.
           noise (positive float, optional): Variance of noise to add to the
               resulting random vector.

        Returns:
            Instance of :class:`gptools.normal.Normal`.
        """

        return Normal(B.reg(self.kernel(x), diag=noise, clip=False),
                      self.mean(x))

    def condition(self, x, y, noise=None):
        """Condition the GP on a number of points.

        Args:
            x (design matrix): Locations of the points to condition on.
            y (design matrix): Values of the points to condition on.
            noise (positive float, optional): Variance of noise to add to the
                resulting random process.

        Returns:
            Instance of :class:`.random.GP`.
        """
        K = Dense(B.reg(self.kernel(x), diag=noise, clip=False))
        return GP(PosteriorKernel(self, x, K), PosteriorMean(self, x, K, y))

    def predict(self, x):
        """Predict at specified locations.

        Args:
            x (design matrix): Locations of the points to predict for.

        Returns:
            A tuple containing the predictive means and predictive marginal
            standard deviations.
        """
        dist = self(x)
        return dist.mean, B.diag(dist.var.mat) ** .5

    @dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a GP and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __add__(self, other):
        return GP(self.kernel, self.mean + other)

    @dispatch(Self)
    def __add__(self, other):
        return Normal(self.kernel + other.kernel, self.mean + other.mean)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a GP and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __mul__(self, other):
        raise NotImplementedError
