# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from fdm import central_fdm
from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter, PromisedType

from .cache import Cache, uprank
from .input import Input, At, MultiInput
from .kernel import ZeroKernel, PosteriorCrossKernel, Kernel, FunctionKernel
from .lazy import LazyVector, LazyMatrix
from .mean import PosteriorCrossMean, Mean
from .mokernel import MultiOutputKernel as MOK
from .momean import MultiOutputMean as MOM
from .random import GPPrimitive, Random
from .spd import SPD

__all__ = ['GP', 'model', 'Graph']

log = logging.getLogger(__name__)

PromisedGP = PromisedType()


class Graph(Referentiable):
    """A GP model."""
    _dispatch = Dispatcher(in_class=Self)

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
            p (:class:`.graph.GP`): GP object to add.
            kernel (:class:`.kernel.Kernel`): Kernel function of GP.
            mean (:class:`.mean.Mean`): Mean function of GP.

        Returns:
            :class:`.graph.GP`: The newly added independent GP.
        """
        # Update means.
        self.means[p] = mean
        # Add rule to kernels.
        self.kernels[p] = kernel
        self.kernels.add_rule((p, None), self.pids, lambda pi: ZeroKernel())
        self.kernels.add_rule((None, p), self.pids, lambda pi: ZeroKernel())
        self._add_p(p)
        return p

    @_dispatch(object, PromisedGP)
    def sum(self, other, p):
        """Sum a GP from the graph with another object.

        Args:
            obj1 (other type or :class:`.graph.GP`): First term in the sum.
            obj2 (other type or :class:`.graph.GP`): Second term in the sum.

        Returns:
            :class:`.graph.GP`: The GP corresponding to the sum.
        """
        return self.sum(p, other)

    @_dispatch(PromisedGP, object)
    def sum(self, p, other):
        kernels = self.kernels  # Careful with the closure!
        return self._update(self.means[p] + other,
                            lambda: kernels[p],
                            lambda pi: kernels[p, pi])

    @_dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        kernels = self.kernels  # Careful with the closure!
        return self._update(self.means[p1] + self.means[p2],
                            (lambda: kernels[p1] + kernels[p2] +
                                     kernels[p1, p2] + kernels[p2, p1]),
                            lambda pi: kernels[p1, pi] + kernels[p2, pi])

    @_dispatch(PromisedGP, B.Numeric)
    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (:class:`.graph.GP`): GP in the product.
            other (object): Other object in the product.

        Returns:
            :class:`.graph.GP`: The GP corresponding to the product.
        """
        kernels = self.kernels  # Careful with the closure!
        return self._update(other * self.means[p],
                            lambda: other ** 2 * kernels[p],
                            lambda pi: other * kernels[p, pi])

    @_dispatch(PromisedGP, FunctionType)
    def mul(self, p, f):
        kernels = self.kernels  # Careful with the closure!

        def ones(x):
            return B.ones([B.shape(x)[0], 1], dtype=B.dtype(x))

        return self._update(f * self.means[p],
                            lambda: f * kernels[p],
                            (lambda pi: FunctionKernel(f, ones) *
                                        kernels[p, pi]))

    @_dispatch(PromisedGP, PromisedGP)
    def mul(self, p1, p2):
        # Careful with the closures!
        kernels = self.kernels
        means = self.means

        def ones(x):
            return B.ones([B.shape(x)[0], 1], dtype=B.dtype(x))

        return self._update(
            (lambda x, B: kernels[p1, p2].elwise(x, x, B)) +
            means[p1] * means[p2],
            (lambda: (kernels[p1] + FunctionKernel(means[p1], means[p1])) *
                     (kernels[p2] + FunctionKernel(means[p2], means[p2])) +
                     (kernels[p1, p2] + FunctionKernel(means[p1], means[p2])) *
                     (kernels[p2, p1] + FunctionKernel(means[p2], means[p1])) -
                     2 * FunctionKernel(means[p1] * means[p2],
                                        means[p1] * means[p2])),
            (lambda pi: FunctionKernel(means[p2], ones) * kernels[p1, pi] +
                        FunctionKernel(means[p1], ones) * kernels[p2, pi]
             ),
        )

    def shift(self, p, shift):
        """Shift a GP.

        Args:
            p (:class:`.graph.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            :class:`.graph.GP`: The shifted GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].shift(shift),
                            lambda: kernels[p].shift(shift),
                            lambda pi: kernels[p, pi].shift(shift, 0))

    def stretch(self, p, stretch):
        """Stretch a GP.

        Args:
            p (:class:`.graph.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            :class:`.graph.GP`: The stretched GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].stretch(stretch),
                            lambda: kernels[p].stretch(stretch),
                            lambda pi: kernels[p, pi].stretch(stretch, 1))

    def select(self, p, *dims):
        """Select input dimensions.

        Args:
            p (:class:`.graph.GP`): GP to select input
                dimensions from.
            *dims (object): Dimensions to select.

        Returns:
            :class:`.graph.GP`: GP with the specific input dimensions.
        """
        kernels = self.kernels
        return self._update(self.means[p].select(dims),
                            lambda: kernels[p].select(dims),
                            lambda pi: kernels[p, pi].select(dims, None))

    def transform(self, p, f):
        """Transform the inputs of a GP.

        Args:
            p (:class:`.graph.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            :class:`.graph.GP`: Input-transformed GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].transform(f),
                            lambda: kernels[p].transform(f),
                            lambda pi: kernels[p, pi].transform(f, None))

    def diff(self, p, dim=0):
        """Differentiate a GP.

        Args:
            p (:class:`.graph.GP`): GP to differentiate.
            dim (int, optional): Dimension of feature which to take the
                derivative with respect to. Defaults to `0`.

        Returns:
            :class:`.graph.GP`: Derivative of GP.
        """
        kernels = self.kernels
        return self._update(self.means[p].diff(dim),
                            lambda: kernels[p].diff(dim),
                            lambda pi: kernels[p, pi].diff(dim, None))

    @_dispatch(At, B.Numeric)
    def condition(self, x, y):
        """Condition the graph on data.

        Can alternatively call this method with tuples `(x, y)` or lists
        `[x, y]`.

        Args:
            x (input): Locations of points to condition on.
            y (tensor): Observations to condition on.
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

    @_dispatch([{tuple, list}])
    def condition(self, *pairs):
        if not all([isinstance(x, At) for x, y in pairs]):
            raise ValueError('Must explicitly specify the processes which to '
                             'condition on.')

        # Extend the graph by the Cartesian product `p` of all processes.
        p = self.cross(*self.ps)

        # Condition the newly created vector-valued GP.
        xs, ys = zip(*pairs)
        y = B.concat([uprank(y) for y in ys], axis=0)
        self.condition(At(p)(MultiInput(*xs)), y)

    def cross(self, *ps):
        """Construct the Cartesian product of a collection of processes.

        Args:
            *ps (:class:`.graph.GP`): Processes to construct the
                Cartesian product of.

        Returns:
            :class:`.graph.GP`: The Cartesian product of `ps`.
        """
        mok = MOK(*ps)
        return self._update(MOM(*ps),
                            lambda: mok,
                            lambda pi: mok.transform(None, lambda y: At(pi)(y)))

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

    @_dispatch(int, [At])
    def sample(self, n, *xs):
        """Sample multiple processes simultaneously.

        Args:
            n (int, optional): Number of samples. Defaults to `1`.
            *xs (:class:`.graph.At`): Locations to sample at.

        Returns:
            tuple: Tuple of samples.
        """
        sample = GPPrimitive(MOK(*self.ps),
                             MOM(*self.ps))(MultiInput(*xs)).sample(n)

        # To unpack `x`, just keep `.get()`ing.
        def unpack(x):
            while isinstance(x, (At, Input)):
                x = x.get()
            return x

        # Unpack sample.
        lengths = [B.shape(uprank(unpack(x)))[0] for x in xs]
        i, samples = 0, []
        for length in lengths:
            samples.append(sample[i:i + length, :])
            i += length
        return samples[0] if len(samples) == 1 else samples

    @_dispatch([At])
    def sample(self, *xs):
        return self.sample(1, *xs)


class GraphMean(Mean, Referentiable):
    """Mean that evaluates to the right mean for a GP attached to a graph.

    Args:
        graph (:class:`.graph.Graph`): Corresponding graph.
        p (:class:`.graph.GP`): Corresponding GP object.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @_dispatch(object)
    def __call__(self, x):
        return self.graph.means[self.p](x)

    @_dispatch(object, Cache)
    def __call__(self, x, cache):
        return self.graph.means[self.p](x, cache)

    @_dispatch(At)
    def __call__(self, x):
        return self.graph.means[type_parameter(x)](x.get())

    @_dispatch(At, Cache)
    def __call__(self, x, cache):
        return self.graph.means[type_parameter(x)](x.get(), cache)

    def __str__(self):
        return str(self.graph.means[self.p])

    @property
    def num_terms(self):
        """Number of terms"""
        return self.graph.means[self.p].num_terms

    def term(self, i):
        """Get a specific term.

        Args:
            i (int): Index of term.

        Returns:
            :class:`.mean.Mean`: The referenced term.
        """
        return self.graph.means[self.p].term(i)

    @property
    def num_factors(self):
        """Number of factors"""
        return self.graph.means[self.p].num_factors

    def factor(self, i):
        """Get a specific factor.

        Args:
            i (int): Index of factor.

        Returns:
            :class:`.mean.Mean`: The referenced factor.
        """
        return self.graph.means[self.p].factor(i)


class GraphKernel(Kernel, Referentiable):
    """Kernel that evaluates to the right kernel for a GP attached to a graph.

    Args:
        graph (:class:`.graph.Graph`): Corresponding graph.
        p (:class:`.graph.GP`): Corresponding GP object.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, graph, p):
        self.p = p
        self.graph = graph

    @_dispatch(object)
    def __call__(self, x):
        return self(x, x)

    @_dispatch.multi((object, Cache), (At, Cache))
    def __call__(self, x, cache):
        return self(x, x, cache)

    @_dispatch(object, object)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, self.p](x, y)

    @_dispatch(object, object, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[self.p, self.p](x, y, cache)

    @_dispatch(object, At)
    def __call__(self, x, y):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get())

    @_dispatch(object, At, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[self.p, type_parameter(y)](x, y.get(), cache)

    @_dispatch(At, object)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y)

    @_dispatch(At, object, Cache)
    def __call__(self, x, y, cache=None):
        return self.graph.kernels[type_parameter(x), self.p](x.get(), y, cache)

    @_dispatch(At, At)
    def __call__(self, x, y):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get())

    @_dispatch(At, At, Cache)
    def __call__(self, x, y, cache):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)](x.get(), y.get(), cache)

    @_dispatch(object)
    def elwise(self, x):
        return self.elwise(x, x)

    @_dispatch.multi((object, Cache), (At, Cache))
    def elwise(self, x, cache):
        return self.elwise(x, x, cache)

    @_dispatch(object, object)
    def elwise(self, x, y):
        return self.graph.kernels[self.p, self.p].elwise(x, y)

    @_dispatch(object, object, Cache)
    def elwise(self, x, y, cache):
        return self.graph.kernels[self.p, self.p].elwise(x, y, cache)

    @_dispatch(object, At)
    def elwise(self, x, y):
        return self.graph.kernels[self.p, type_parameter(y)].elwise(x, y.get())

    @_dispatch(object, At, Cache)
    def elwise(self, x, y, cache):
        return self.graph.kernels[self.p,
                                  type_parameter(y)].elwise(x, y.get(), cache)

    @_dispatch(At, object)
    def elwise(self, x, y):
        return self.graph.kernels[type_parameter(x), self.p].elwise(x.get(), y)

    @_dispatch(At, object, Cache)
    def elwise(self, x, y, cache=None):
        return self.graph.kernels[type_parameter(x),
                                  self.p].elwise(x.get(), y, cache)

    @_dispatch(At, At)
    def elwise(self, x, y):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)].elwise(x.get(), y.get())

    @_dispatch(At, At, Cache)
    def elwise(self, x, y, cache):
        return self.graph.kernels[type_parameter(x),
                                  type_parameter(y)].elwise(x.get(),
                                                            y.get(), cache)

    @property
    def stationary(self):
        """Stationarity of the kernel."""
        return self.graph.kernels[self.p].stationary

    @property
    def var(self):
        """Variance of the kernel."""
        return self.graph.kernels[self.p].var

    @property
    def length_scale(self):
        """Approximation of the length scale of the kernel."""
        return self.graph.kernels[self.p].length_scale

    @property
    def period(self):
        """Period of the kernel."""
        return self.graph.kernels[self.p].period

    def __str__(self):
        return str(self.graph.kernels[self.p])

    @property
    def num_terms(self):
        """Number of terms"""
        return self.graph.kernels[self.p].num_terms

    def term(self, i):
        """Get a specific term.

        Args:
            i (int): Index of term.

        Returns:
            :class:`.kernel.Kernel`: The referenced term.
        """
        return self.graph.kernels[self.p].term(i)

    @property
    def num_factors(self):
        """Number of factors"""
        return self.graph.kernels[self.p].num_factors

    def factor(self, i):
        """Get a specific factor.

        Args:
            i (int): Index of factor.

        Returns:
            :class:`.kernel.Kernel`: The referenced factor.
        """
        return self.graph.kernels[self.p].factor(i)


model = Graph()  #: A default graph provided for convenience


class GP(GPPrimitive, Referentiable):
    """Gaussian process.

    Args:
        kernel (:class:`.kernel.Kernel`): Kernel of the
            process.
        mean (:class:`.mean.Mean`, optional): Mean function of the
            process. Defaults to zero.
        graph (:class:`.graph.Graph`, optional): Graph to attach to.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch([object])
    def __init__(self, kernel, mean=None, graph=model):
        # First resolve `kernel` and `mean` through `GPPrimitive`s constructor.
        GPPrimitive.__init__(self, kernel, mean)
        kernel, mean = self.kernel, self.mean

        # Then add a new `GP` to the graph.
        GP.__init__(self, graph)
        graph.add_independent_gp(self, kernel, mean)

    @_dispatch(Graph)
    def __init__(self, graph):
        GPPrimitive.__init__(
            self,
            GraphKernel(graph, self),
            GraphMean(graph, self)
        )
        self.graph = graph

    @_dispatch(object)
    def __add__(self, other):
        return self.graph.sum(self, other)

    @_dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a GP and a {}.'
                                  ''.format(type(other).__name__))

    @_dispatch(Self)
    def __add__(self, other):
        if self.graph != other.graph:
            raise RuntimeError('Can only add GPs from the same graph.')
        return self.graph.sum(self, other)

    @_dispatch(object)
    def __mul__(self, other):
        return self.graph.mul(self, other)

    @_dispatch(Random)
    def __mul__(self, other):
        raise NotImplementedError('Cannot multiply a GP and a {}.'
                                  ''.format(type(other).__name__))

    @_dispatch(Self)
    def __mul__(self, other):
        return self.graph.mul(self, other)

    @_dispatch(At, B.Numeric)
    def condition(self, x, y):
        """Condition the GP. See :meth:`.graph.Graph.condition`."""
        self.graph.condition(x, y)
        return self

    @_dispatch({B.Numeric, Input}, B.Numeric)
    def condition(self, x, y):
        self.graph.condition(At(self)(x), y)
        return self

    @_dispatch([{tuple, list}])
    def condition(self, *pairs):
        # Set any unspecified locations to this process.
        pairs = [(x, y) if isinstance(x, At) else (At(self)(x), y)
                 for x, y in pairs]
        self.graph.condition(*pairs)
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
        """Shift the GP. See :meth:`.graph.Graph.shift`."""
        return self.graph.shift(self, shift)

    def stretch(self, stretch):
        """Stretch the GP. See :meth:`.graph.Graph.stretch`."""
        return self.graph.stretch(self, stretch)

    def transform(self, f):
        """Input transform the GP. See :meth:`.graph.Graph.transform`."""
        return self.graph.transform(self, f)

    def select(self, *dims):
        """Select dimensions from the input. See :meth:`.graph.Graph.select`."""
        return self.graph.select(self, *dims)

    def diff(self, dim=0):
        """Differentiate the GP. See :meth:`.graph.Graph.diff`."""
        return self.graph.diff(self, dim)

    def diff_approx(self, deriv=1, order=5, eps=1e-8, bound=1.):
        """Approximate the derivative of the GP by constructing a finite
        difference approximation.

        Args:
            deriv (int): Order of the derivative.
            order (int): Order of the estimate.
            eps (float, optional): Absolute round-off error of the function
                evaluation. This is used to estimate the step size.
            bound (float, optional): Upper bound of the absolute value of the
                function and all its derivatives. This is used to estimate
                the step size.

        Returns:
            Approximation of the derivative of the GP.
        """
        # Use the FDM library to figure out the coefficients.
        fdm = central_fdm(order, deriv, eps=eps, bound=bound)

        # Construct finite difference.
        df = 0
        for g, c in zip(fdm.grid, fdm.coefs):
            df += c * self.shift(-g * fdm.step)
        return df / fdm.step ** deriv


PromisedGP.deliver(GP)
