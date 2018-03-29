# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable

from stheno import SPD, ZeroMean, PosteriorMean, PosteriorKernel

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
        var (matrix or instance of :class:`.pdmat.PDMat`): Variance of the
            distribution.
        mean (column vector, optional): Mean of the distribution, defaults to
            zero.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, var, mean=None):
        self.dim = B.shape(var)[-1]
        # TODO: proper zeros here
        self.mean = 0 if mean is None else mean
        self.var = var if isinstance(var, SPD) else SPD(var)
        self.dtype = self.var.mat.dtype

    def m2(self):
        """Second moment of the distribution"""
        return self.var + B.outer(self.mean)

    def log_pdf(self, x):
        """Compute the log-pdf.

        Args:
            x (design matrix): Values to compute the log-pdf of.
        """
        if B.rank(x) != 2:
            raise ValueError('IBut must have rank 2.')
        if B.shape(x)[0] != self.dim:
            raise RuntimeError('Dimensionality of data points does not match '
                               'that of the distribution.')
        n = B.shape(x)[1]  # Number of data points
        return -(self.var.log_det() +
                 n * self.dim * B.cast(B.log_2_pi, dtype=self.dtype) +
                 self.var.mah_dist2(x - self.mean)) / 2

    def kl(self, other):
        """Compute the KL divergence with respect to another normal
        distribution.

        Args:
            other (instance of :class:`.random.Normal`): Other normal.
        """
        if self.dim != other.dim:
            raise RuntimeError('KL divergence can only be computed between '
                               'distributions of the same dimensionality.')
        return (self.var.ratio(other.var) +
                other.var.mah_dist2(other.mean - self.mean) - self.dim +
                other.var.log_det() - self.var.log_det()) / 2

    def w2(self, other):
        """Compute the 2-Wasserstein distance with respect to another normal
        distribution.

        Args:
            other (instance of :class:`.random.Normal`): Other normal.
        """
        if self.dim != other.dim:
            raise RuntimeError('W2 distance can only be computed between '
                               'distributions of the same dimensionality.')
        root = SPD(B.dot(B.dot(self.var.root(), other.var.mat),
                          self.var.root())).root()
        var_part = B.trace(self.var.mat) + B.trace(other.var.mat) - \
                   2 * B.trace(root)
        mean_part = B.sum((self.mean - other.mean) ** 2)
        # The sum of `mean_part` and `var_par` should be positive, but this
        # may not be the case due to numerical errors.
        return B.abs(mean_part + var_part) ** .5

    def sample(self, num=1, noise=None):
        """Sample form the distribution.

        Args:
            num (int): Number of samples.
            noise (positive float, optional): Variance of noise to add to the
                samples.
        """
        if noise is None or noise == 0:
            L = self.var.cholesky()
        else:
            L = B.cholesky(B.reg(self.var.mat, diag=noise))
        e = B.randn((self.dim, num), dtype=L.dtype)
        return B.dot(L, e) + self.mean

    @dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __add__(self, other):
        return Normal(self.var, self.mean + other)

    @dispatch(Self)
    def __add__(self, other):
        return Normal(self.var.mat + other.var.mat, self.mean + other.mean)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(Random)
    def __rmul__(self, other):
        raise NotImplementedError('Cannot multiply a Normal and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(object)
    def __rmul__(self, other):
        return Normal(B.dot(B.dot(other, self.var.mat), other, tr_b=True),
                      B.dot(other, self.mean))

    @dispatch(object)
    def __mul__(self, other):
        return Normal(B.dot(B.dot(other, self.var.mat, tr_a=True), other),
                      B.dot(other, self.mean, tr_a=True))


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
        K = SPD(B.reg(self.kernel(x), diag=noise, clip=False))
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
        return Normal(UnimplementedOperation, other * self.mean)
