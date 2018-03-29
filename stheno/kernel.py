# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable

__all__ = ['Kernel', 'StationaryKernel', 'EQ', 'Matern12', 'Exp', 'Matern32',
           'Matern52', 'RQ', 'Noise', 'Linear', 'PosteriorKernel']


class Kernel(Referentiable):
    """Kernel function.

    Kernels can be added and multiplied.

    Args:
        f (function): Function that implements the kernel. It should take in
            two design matrices and return the resulting kernel matrix.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, f):
        self.f = f

    def __call__(self, x, y=None):
        """Construct the kernel for design matrices of points.

        Args:
            x (design matrix): First argument.
            y (design matrix, optional): Second argument. Defaults to first
                argument.

        Returns:
            Kernel matrix.
        """
        y = x if y is None else y
        if B.rank(x) != 2 or B.rank(y) != 2:
            raise ValueError('Arguments must have rank 2.')
        return self.f(x, y)

    @dispatch(Self)
    def __add__(self, other):
        return Kernel(lambda *args: self(*args) + other(*args))

    @dispatch(object)
    def __add__(self, other):
        return Kernel(lambda *args: self(*args) + other)

    def __radd__(self, other):
        return self + other

    @dispatch(Self)
    def __mul__(self, other):
        return Kernel(lambda *args: self(*args) * other(*args))

    @dispatch(object)
    def __mul__(self, other):
        return Kernel(lambda *args: self(*args) * other)

    def __rmul__(self, other):
        return self * other

    def stretch(self, scale):
        """Stretch the kernel.

        Args:
            scale (tensor): Scale.
        """
        return Kernel(lambda x, y: self.f(x / scale, y / scale))

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor, optional): Period. Defaults to `1`.
        """

        def feat_map(x):
            scale = 2 * B.pi / B.cast(period, x.dtype)
            return B.concatenate((B.sin(x * scale),
                                   B.cos(x * scale)), axis=0)

        return Kernel(lambda x, y: self.f(feat_map(x), feat_map(y)))


class StationaryKernel(Kernel, Referentiable):
    """Stationary kernel."""

    dispatch = Dispatcher(in_class=Self)

    def stretch(self, scale):
        return StationaryKernel(Kernel.stretch(self, scale).f)

    def periodic(self, period=1):
        return StationaryKernel(Kernel.periodic(self, period).f)

    @dispatch(Self)
    def __add__(self, other):
        return StationaryKernel(Kernel.__add__(self, other).f)

    @dispatch(Self)
    def __mul__(self, other):
        return StationaryKernel(Kernel.__mul__(self, other).f)


class EQ(StationaryKernel):
    """Exponentiated quadratic kernel."""

    def __init__(self):
        self.f = lambda x, y: B.exp(-.5 * B.pw_dists2(x, y))


class RQ(StationaryKernel):
    """Rational quadratic kernel.

    Args:
        alpha (positive float): Shape of the prior over length scales.
            determines the weight of the tails of the kernel.
    """

    def __init__(self, alpha):
        alpha = B.cast(alpha)
        self.f = lambda x, y: (1 + .5 * B.pw_dists2(x, y) / alpha) ** (-alpha)


class Exp(StationaryKernel):
    """Exponential kernel."""

    def __init__(self):
        self.f = lambda x, y: B.exp(-(B.pw_dists2(x, y) + 1e-6) ** .5)


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(StationaryKernel):
    """Matern--3/2 kernel."""

    def __init__(self):
        def f(x, y):
            r = 3 ** .5 * (B.pw_dists2(x, y) + 1e-6) ** .5
            return (1 + r) * B.exp(-r)

        self.f = f


class Matern52(StationaryKernel):
    """Matern--5/2 kernel."""

    def __init__(self):
        def f(x, y):
            dists2 = B.pw_dists2(x, y)
            r1 = 5 ** .5 * (dists2 + 1e-6) ** .5
            r2 = 5 * dists2 / 3
            return (1 + r1 + r2) * B.exp(-r1)

        self.f = f


class Noise(StationaryKernel):
    """Noise kernel."""

    def __init__(self):
        def f(x, y):
            dists = B.pw_dists2(x, y) ** .5
            return B.cast(dists < 1e-10, x.dtype)

        self.f = f


class Linear(Kernel):
    """Linear kernel."""

    def __init__(self):
        def f(x, y):
            return B.matmul(x, y, tr_a=True)

        self.f = f


class PosteriorKernel(Kernel):
    def __init__(self, gp, z, Kz):
        def f(x, y):
            return (gp.kernel(x, y) - Kz.quadratic_form(gp.kernel(z, x),
                                                        gp.kernel(z, y)))

        self.f = f
