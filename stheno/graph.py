# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable, type_parameter, kind, \
    PromisedType
from lab import B

from .kernel import ZeroKernel, PosteriorCrossKernel, Kernel
from .mean import PosteriorCrossMean, ZeroMean, Mean
from .random import GPPrimitive, Random
from .spd import SPD
from .lazy import LazyVector, LazyMatrix
from .cache import Cache

__all__ = ['GP', 'model', 'Graph', 'At']

At = kind()
PromisedGP = PromisedType()


class Graph(Referentiable):
    """A GP model."""
    dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self.ps = []
        self.pids = set()
        self.kernels = LazyMatrix()
        self.means = LazyVector()
        self.prior_kernels = None
        self.prior_means = None

    def _add_p(self, p):
        self.ps.append(p)
        self.pids.add(id(p))

    def _update(self, mean, k_ii_generator, k_ij_generator):
        p = GP(self)
        self.means[p] = mean
        self.kernels.add_rule((p, p), self.pids, k_ii_generator)
        self.kernels.add_rule((p, None), self.pids, k_ij_generator)
        kernels = self.kernels  # Careful with the closure!
        self.kernels.add_rule((None, p), self.pids,
                              lambda pi: reversed(kernels[p, pi]))
        self._add_p(p)
        return p

    def add_independent_gp(self, p, kernel, mean):
        """Add an independent GP to the model.

        Args:
            p (instance of :class:`.graph.GP`): GP object to add.
            kernel (instance of :class:`.kernel.Kernel`): Kernel function of GP.
            mean (instance of :class:`.mean.Mean`): Mean function of GP.
        """
        # Update means.
        self.means[p] = mean
        # Add rule to kernels.
        self.kernels[p] = kernel
        self.kernels.add_rule((p, None), self.pids, lambda pi: ZeroKernel())
        self.kernels.add_rule((None, p), self.pids, lambda pi: ZeroKernel())
        self._add_p(p)
        return p

    @dispatch(object, PromisedGP)
    def sum(self, other, p):
        """Sum a GP from the graph with another object.

        Args:
            obj1 (other object or instance of :class:`.graph.GP`): First term
                in the sum.
            obj2 (other object or instance of :class:`.graph.GP`): Second term
                in the sum.

        Returns:
            The GP corresponding to the sum.
        """
        return self.sum(p, other)

    @dispatch(PromisedGP, object)
    def sum(self, p, other):
        kernels = self.kernels  # Careful with the closure!
        return self._update(self.means[p] + other,
                            lambda: kernels[p],
                            lambda pi: kernels[p, pi])

    @dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        kernels = self.kernels  # Careful with the closure!
        return self._update(self.means[p1] + self.means[p2],
                            (lambda: kernels[p1] + kernels[p2] +
                                     kernels[p1, p2] + kernels[p2, p1]),
                            lambda pi: kernels[p1, pi] + kernels[p2, pi])

    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (instance of :class:`.graph.GP`): GP in the product.
            other (object): Other object in the product.

        Returns:
            The GP corresponding to the product.
        """
        kernels = self.kernels  # Careful with the closure!
        return self._update(other * self.means[p],
                            lambda: other ** 2 * kernels[p],
                            lambda pi: other * kernels[p, pi])

    def shift(self, p, shift):
        """Shift a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            The shifted GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].shift(shift),
                            lambda: kernels[p].shift(shift),
                            lambda pi: kernels[p, pi].shift(shift, 0))

    def stretch(self, p, stretch):
        """Stretch a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            The stretched GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].stretch(stretch),
                            lambda: kernels[p].stretch(stretch),
                            lambda pi: kernels[p, pi].stretch(stretch, 1))

    def select(self, p, *dims):
        """Select input dimensions.

        Args:
            p (instance of :class:`.graph.GP`): GP to select input
                dimensions from.
            \*dims (object): Dimensions to select.

        Returns:
            GP with the specific input dimensions.
        """
        kernels = self.kernels
        return self._update(self.means[p].select(dims),
                            lambda: kernels[p].select(dims),
                            lambda pi: kernels[p, pi].select(dims, None))

    def transform(self, p, f):
        """Transform the inputs of a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            Input-transformed GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].transform(f),
                            lambda: kernels[p].transform(f),
                            lambda pi: kernels[p, pi].transform(f, None))

    def diff(self, p, deriv=0):
        """Differentiate a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to differentiate.
            deriv (int, optional): Index of feature which to take the
                derivative of. Defaults to `0`.

        Returns:
            Derivative of GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].diff(deriv),
                            lambda: kernels[p].diff(deriv),
                            lambda pi: kernels[p, pi].diff(deriv, None))

    @dispatch(At, object)
    def condition(self, x, y):
        """Condition the graph on data.

        Args:
            x (design matrix): Locations of points to condition on.
            y (design matrix): Observations to condition on.
        """
        p_data, x = type_parameter(x), x.get()
        Kx = SPD(self.kernels[p_data](x))

        prior_kernels = self.kernels
        prior_means = self.means

        # Store prior if it isn't already.
        if self.prior_kernels is None:
            self.prior_kernels = prior_kernels
            self.prior_means = prior_means

        def build_posterior_mean(pi):
            return PosteriorCrossMean(prior_means[pi],
                                      prior_means[p_data],
                                      prior_kernels[p_data, pi],
                                      x, Kx, y)

        def build_posterior_kernel(pi, pj):
            return PosteriorCrossKernel(prior_kernels[pi, pj],
                                        prior_kernels[p_data, pi],
                                        prior_kernels[p_data, pj],
                                        x, Kx)

        # Update to posterior.
        self.kernels = LazyMatrix()
        self.kernels.add_rule((None, None), self.pids, build_posterior_kernel)
        self.means = LazyVector()
        self.means.add_rule((None,), self.pids, build_posterior_mean)

    def revert_prior(self):
        """Revert the model back to the state before any conditioning
        operations.
        """
        if self.prior_kernels is not None:
            # Reverts kernels and means.
            self.kernels = self.prior_kernels
            self.means = self.prior_means

            # Empty storage.
            self.prior_kernels = None
            self.prior_means = None


class GraphMean(Mean, Referentiable):
    """Mean that evaluates to the right mean for a GP attached to a graph.

    Args:
        graph (instance of :class:`.graph.Graph`): Corresponding graph.
        p (instance of :class:`.graph.GP`): Corresponding GP object.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @dispatch(object)
    def __call__(self, x):
        return self.graph.means[self.p](x)

    @dispatch(object, Cache)
    def __call__(self, x, cache):
        return self.graph.means[self.p](x, cache)

    @dispatch(At)
    def __call__(self, x):
        return self.graph.means[type_parameter(x)](x.get())

    @dispatch(At, Cache)
    def __call__(self, x, cache):
        return self.graph.means[type_parameter(x)](x.get(), cache)

    def __str__(self):
        return str(self.graph.means[self.p])


class GraphKernel(Kernel, Referentiable):
    """Kernel that evaluates to the right kernel for a GP attached to a graph.

    Args:
        graph (instance of :class:`.graph.Graph`): Corresponding graph.
        p (instance of :class:`.graph.GP`): Corresponding GP object.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @dispatch(object)
    def __call__(self, x):
        return self(x, x)

    @dispatch.multi((object, Cache), (At, Cache))
    def __call__(self, x, cache):
        return self(x, x, cache)

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, self.p](x, y)

    @dispatch(object, object, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[self.p, self.p](x, y, cache)

    @dispatch(object, At)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get())

    @dispatch(object, At, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get(), cache)

    @dispatch(At, object)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y)

    @dispatch(At, object, Cache)
    def __call__(self, x, y, cache=None):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y, cache)

    @dispatch(At, At)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get())

    @dispatch(At, At, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get(), cache)

    @property
    def stationary(self):
        return self.graph.kernels[self.p].stationary

    @property
    def var(self):
        return self.graph.kernels[self.p].var

    @property
    def length_scale(self):
        return self.graph.kernels[self.p].length_scale

    @property
    def period(self):
        return self.graph.kernels[self.p].period

    def __str__(self):
        return str(self.graph.kernels[self.p])


model = Graph()  #: A default graph provided for convenience


class GP(GPPrimitive, Referentiable):
    """Gaussian process.

    Args:
        kernel (instance of :class:`.kernel.Kernel`): Kernel of the
            process.
        mean (instance of :class:`.mean.Mean`, optional): Mean function of the
            process. Defaults to zero.
        graph (instance of :class:`.graph.Graph`, optional): Graph to attach to.
    """
    dispatch = Dispatcher(in_class=Self)

    @dispatch([object])
    def __init__(self, kernel, mean=None, graph=model):
        # First resolve `kernel` and `mean` through `GPPrimitive`s constructor.
        GPPrimitive.__init__(self, kernel, mean)
        kernel, mean = self.kernel, self.mean

        # Then add a new `GP` to the graph.
        GP.__init__(self, graph)
        graph.add_independent_gp(self, kernel, mean)

    @dispatch(Graph)
    def __init__(self, graph):
        GPPrimitive.__init__(
            self,
            GraphKernel(graph, self),
            GraphMean(graph, self)
        )
        self.graph = graph

    @dispatch(object)
    def __add__(self, other):
        return self.graph.sum(self, other)

    @dispatch(Self)
    def __add__(self, other):
        if self.graph != other.graph:
            raise RuntimeError('Can only add GPs from the same graph.')
        return self.graph.sum(self, other)

    @dispatch(object)
    def __mul__(self, other):
        return self.graph.mul(self, other)

    @dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a GP and a {}.'
                                  ''.format(type(other).__name__))

    @dispatch(At, object)
    def condition(self, x, y):
        """Condition the GP """
        self.graph.condition(x, y)
        return self

    @dispatch(object, object)
    def condition(self, x, y):
        self.graph.condition(At(self)(x), y)
        return self

    def __matmul__(self, other):
        """Alternative to writing `At(self)(other)`"""
        return At(self)(other)

    def revert_prior(self):
        """Revert the model back to the state before any conditioning
        operations.
        """
        self.graph.revert_prior()

    def shift(self, shift):
        return self.graph.shift(self, shift)

    def stretch(self, stretch):
        return self.graph.stretch(self, stretch)

    def transform(self, f):
        return self.graph.transform(self, f)

    def select(self, *dims):
        return self.graph.select(self, *dims)

    def diff(self, deriv=0):
        return self.graph.diff(self, deriv)


PromisedGP.deliver(GP)
