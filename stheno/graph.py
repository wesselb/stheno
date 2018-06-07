# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable, type_parameter, kind, \
    PromisedType

from .kernel import ZeroKernel, PosteriorCrossKernel, Kernel
from .mean import PosteriorCrossMean, ZeroMean, Mean
from .random import GPPrimitive, Random
from .spd import SPD
from .lazy import LazyVector, LazySymmetricMatrix

__all__ = ['GP', 'model', 'Graph', 'At']

At = kind()
PromisedGP = PromisedType()


class Graph(Referentiable):
    """A GP model."""
    dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self.ps = []
        self.pids = set()
        self.kernels = LazySymmetricMatrix()
        self.means = LazyVector()
        self.prior_kernels = None
        self.prior_means = None

    def _add_p(self, p):
        self.ps.append(p)
        self.pids.add(id(p))

    def add_independent_gp(self, p, kernel, mean):
        """Add an independent GP to the model.

        Args:
            p (instance of :class:`.graph.GP`): GP object to add.
            kernel (instance of :class:`.kernel.Kernel`): Kernel function of GP.
            mean (instance of :class:`.mean.Mean`): Mean function of GP.
        """
        mean = ZeroMean() if mean is None else mean
        # Update means.
        self.means[p] = mean
        # Add rule to kernels.
        self.kernels[p] = kernel
        self.kernels.add_rule((p, None), self.pids, lambda pi: ZeroKernel())
        self._add_p(p)

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
        p_sum = GP(self)
        self._add_p(p_sum)
        # Update means.
        self.means[p_sum] = self.means[p] + other
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_sum, None), self.pids,
                              lambda pi: kernels[p, pi])
        return p_sum

    @dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        p_sum = GP(self)
        self._add_p(p_sum)
        # Update means.
        self.means[p_sum] = self.means[p1] + self.means[p2]
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_sum, None), self.pids,
                              lambda pi: kernels[p1, pi] + kernels[p2, pi])
        return p_sum

    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (instance of :class:`.graph.GP`): GP in the product.
            other (obj): Other object in the product.

        Returns:
            The GP corresponding to the product.
        """
        p_prod = GP(self)
        # Update means.
        self.means[p_prod] = other * self.means[p]
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_prod, p_prod), self.pids,
                              lambda: other ** 2 * kernels[p])
        self.kernels.add_rule((p_prod, None), self.pids,
                              lambda pi: other * kernels[p, pi])
        self._add_p(p_prod)
        return p_prod

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
        self.kernels = LazySymmetricMatrix()
        self.kernels.add_rule((None, None), self.pids, build_posterior_kernel)
        self.means = LazyVector()
        self.means.add_rule((None,), self.pids, build_posterior_mean)

    def revert_prior(self):
        """Revert the model back to the state before any conditioning
        operations.
        """
        if self.prior_kernels is not None:
            self.kernels = self.prior_kernels
            self.means = self.prior_means


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

    @dispatch(At)
    def __call__(self, x):
        return self.graph.means[type_parameter(x)](x.get())

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

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, self.p](x, y)

    @dispatch(object, At)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get())

    @dispatch(At, object)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y)

    @dispatch(At, At)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get())

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

    @property
    def primitive(self):
        return self.graph.kernels[self.p].primitive

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


PromisedGP.deliver(GP)
