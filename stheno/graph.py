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
        # Update means.
        self.means[p_sum] = self.means[p] + other
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_sum, p_sum), self.pids, lambda: kernels[p, p])
        self.kernels.add_rule((p_sum, None), self.pids,
                              lambda pi: kernels[p, pi])
        self.kernels.add_rule((None, p_sum), self.pids,
                              lambda pi: reversed(kernels[p_sum, pi]))
        self._add_p(p_sum)
        return p_sum

    @dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        p_sum = GP(self)
        # Update means.
        self.means[p_sum] = self.means[p1] + self.means[p2]
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_sum, p_sum), self.pids,
                              lambda: kernels[p1] + kernels[p2] +
                                      2 * kernels[p1, p2])
        self.kernels.add_rule((p_sum, None), self.pids,
                              lambda pi: kernels[p1, pi] + kernels[p2, pi])
        self.kernels.add_rule((None, p_sum), self.pids,
                              lambda pi: reversed(kernels[p_sum, pi]))
        self._add_p(p_sum)
        return p_sum

    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (instance of :class:`.graph.GP`): GP in the product.
            other (object): Other object in the product.

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
        self.kernels.add_rule((None, p_prod), self.pids,
                              lambda pi: reversed(kernels[p_prod, pi]))
        self._add_p(p_prod)
        return p_prod

    def shift(self, p, shift):
        """Shift a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            The shifted GP.
        """
        p_shifted = GP(self)
        # Update means.
        self.means[p_shifted] = self.means[p].shift(shift)
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_shifted, p_shifted), self.pids,
                              lambda: kernels[p].shift(shift))
        self.kernels.add_rule((p_shifted, None), self.pids,
                              lambda pi: kernels[p, pi].shift(shift, 0))
        self.kernels.add_rule((None, p_shifted), self.pids,
                              lambda pi: reversed(kernels[p_shifted, pi]))
        self._add_p(p_shifted)
        return p_shifted

    def stretch(self, p, stretch):
        """Stretch a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            The stretched GP.
        """
        p_stretched = GP(self)
        # Update means.
        self.means[p_stretched] = self.means[p].stretch(stretch)
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_stretched, p_stretched), self.pids,
                              lambda: kernels[p].stretch(stretch))
        self.kernels.add_rule((p_stretched, None), self.pids,
                              lambda pi: kernels[p, pi].stretch(stretch, 1))
        self.kernels.add_rule((None, p_stretched), self.pids,
                              lambda pi: reversed(kernels[p_stretched, pi]))
        self._add_p(p_stretched)
        return p_stretched

    def select(self, p, *dims):
        """Select input dimensions.

        Args:
            p (instance of :class:`.graph.GP`): GP to select input
                dimensions from.
            \*dims (object): Dimensions to select.

        Returns:
            GP with the specific input dimensions.
        """
        p_select = GP(self)
        # Update means.
        self.means[p_select] = self.means[p].select(dims)
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_select, p_select), self.pids,
                              lambda: kernels[p].select(dims))
        self.kernels.add_rule((p_select, None), self.pids,
                              lambda pi: kernels[p, pi].select(dims, None))
        self.kernels.add_rule((None, p_select), self.pids,
                              lambda pi: reversed(kernels[p_select, pi]))
        self._add_p(p_select)
        return p_select

    def transform(self, p, f):
        """Transform the inputs of a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            Input-transformed GP.
        """
        p_transformed = GP(self)
        # Update means.
        self.means[p_transformed] = self.means[p].transform(f)
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((p_transformed, p_transformed), self.pids,
                              lambda: kernels[p].transform(f))
        self.kernels.add_rule((p_transformed, None), self.pids,
                              lambda pi: kernels[p, pi].transform(
                                  f, (lambda x, B: x)
                              ))
        self.kernels.add_rule((None, p_transformed), self.pids,
                              lambda pi: reversed(kernels[p_transformed, pi]))
        self._add_p(p_transformed)
        return p_transformed

    def diff(self, p, deriv=0):
        """Differentiate a GP.

        Args:
            p (instance of :class:`.graph.GP`): GP to differentiate.
            deriv (int, optional): Index of feature which to take the
                derivative of. Defaults to `0`.

        Returns:
            Derivative of GP.
        """
        dp = GP(self)
        # Update means.
        self.means[dp] = self.means[p].diff(deriv)
        # Add rule to kernels.
        kernels = self.kernels
        self.kernels.add_rule((dp, dp), self.pids,
                              lambda: kernels[p].diff(deriv))
        self.kernels.add_rule((dp, None), self.pids,
                              lambda pi: kernels[p, pi].diff(deriv, None))
        self.kernels.add_rule((None, dp), self.pids,
                              lambda pi: reversed(kernels[dp, pi]))
        self._add_p(dp)
        return dp

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
