# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from fdm import central_fdm
from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter, PromisedType

from .cache import Cache, uprank
from .field import Formatter
from .input import Input, At, MultiInput
from .kernel import ZeroKernel, PosteriorKernel, Kernel, \
    TensorProductKernel, CorrectiveKernel, Delta
from .lazy import LazyVector, LazyMatrix
from .matrix import matrix, Diagonal
from .mean import PosteriorMean, Mean
from .mokernel import MultiOutputKernel as MOK
from .momean import MultiOutputMean as MOM
from .random import GPPrimitive, Random

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

        # Create storage for prior kernels and means.
        self.prior_kernels = None
        self.prior_means = None

        # Store named GPs in both ways.
        self.gps_by_name = {}
        self.names_by_gp = {}

    @_dispatch(str)
    def __getitem__(self, name):
        return self.gps_by_name[name]

    @_dispatch(PromisedGP)
    def __getitem__(self, p):
        return self.names_by_gp[id(p)]

    @_dispatch(PromisedGP, str)
    def name(self, p, name):
        """Name a GP.

        Args:
            p (:class:`.graph.GP`): GP to name.
            name (str): Name. Must be unique.
        """
        # Delete any existing names and back-references for the GP.
        if id(p) in self.names_by_gp:
            del self.gps_by_name[self.names_by_gp[id(p)]]
            del self.names_by_gp[id(p)]

        # Check that name is not in use.
        if name in self.gps_by_name:
            raise RuntimeError('Name "{}" for "{}" already taken by "{}".'
                               ''.format(name, p, self[name]))

        # Set the name and the back-reference.
        self.gps_by_name[name] = p
        self.names_by_gp[id(p)] = name

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
        # Check that the GPs are on the same graph.
        if p1.graph != p2.graph:
            raise RuntimeError('Can only add GPs from the same graph.')

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
        return self._update(self.means[p] * other,
                            lambda: kernels[p] * other ** 2,
                            lambda pi: kernels[p, pi] * other)

    @_dispatch(PromisedGP, FunctionType)
    def mul(self, p, f):
        kernels = self.kernels  # Careful with the closure!

        def ones(x):
            return B.ones([B.shape(x)[0], 1], dtype=B.dtype(x))

        return self._update(f * self.means[p],
                            lambda: f * kernels[p],
                            (lambda pi: TensorProductKernel(f, ones) *
                                        kernels[p, pi]))

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
        p_x, x = type_parameter(x), x.get()
        K_x = matrix(self.kernels[p_x](x))

        # Careful with the closure!
        kernels = self.kernels
        means = self.means

        def build_posterior_kernel(pi, pj):
            return PosteriorKernel(
                kernels[pi, pj], kernels[p_x, pi], kernels[p_x, pj], x, K_x
            )

        def build_posterior_mean(pi):
            return PosteriorMean(
                means[pi], means[p_x], kernels[p_x, pi], x, K_x, y
            )

        self.condition(build_posterior_kernel, build_posterior_mean)

    @_dispatch(At, B.Numeric, At, PromisedGP)
    def condition(self, x, y, z, e):
        """Sparsely condition the graph on data.

        Args:
            x (input): Locations of points to condition on.
            y (tensor): Observations to condition on.
            z (input): Locations of inducing points.
            e (:class:`.graph.GP`): GP representing the additive, independent
                noise process.
        """
        K_z, L_z, K_zx, K_n, A, y_bar, prod_y_bar = \
            self._sparse_components(x, y, z, e)
        p_z, z = type_parameter(z), z.get()

        # Compute the optimal mean.
        mu = self.means[p_z](z) + B.qf(A, B.trisolve(L_z, K_z), prod_y_bar)

        # Careful with the closure!
        kernels = self.kernels
        means = self.means

        def build_posterior_kernel(pi, pj):
            return PosteriorKernel(
                kernels[pi, pj], kernels[p_z, pi], kernels[p_z, pj], z, K_z
            ) + CorrectiveKernel(
                kernels[p_z, pi], kernels[p_z, pj], z, A, K_z
            )

        def build_posterior_mean(pi):
            return PosteriorMean(
                means[pi], means[p_z], kernels[p_z, pi], z, K_z, mu
            )

        self.condition(build_posterior_kernel, build_posterior_mean)

    @_dispatch(At, B.Numeric, At, PromisedGP)
    def elbo(self, x, y, z, e):
        """Compute the ELBO.

        Args:
            x (input): Locations of points to condition on.
            y (tensor): Observations to condition on.
            z (input): Locations of inducing points.
            e (:class:`.graph.GP`): GP representing the additive, independent
                noise process.

        Returns:
            tensor: ELBO.
        """
        K_z, L_z, K_zx, K_n, A, y_bar, prod_y_bar = \
            self._sparse_components(x, y, z, e)
        p_x, x = type_parameter(x), x.get()

        # Compute the ELBO.
        trace_part = B.ratio(Diagonal(self.kernels[p_x].elwise(x)[:, 0]) -
                             Diagonal(B.qf_diag(K_z, K_zx)), K_n)
        det_part = B.logdet(2 * B.pi * K_n) + B.logdet(A)
        qf_part = B.qf(K_n, y_bar)[0, 0] - B.qf(A, prod_y_bar)[0, 0]
        return -0.5 * (trace_part + det_part + qf_part)

    @_dispatch(At, B.Numeric, At, PromisedGP)
    def _sparse_components(self, x, y, z, e):
        p_x, x = type_parameter(x), x.get()
        p_z, z = type_parameter(z), z.get()

        # TODO: Get `e` via graph analysis.

        # Construct the necessary kernel matrices.
        K_zx = self.kernels[p_z, p_x](z, x)
        K_z = self.kernels[p_z](z)
        K_n = e.kernel(x)

        if not isinstance(K_n, Diagonal):
            raise RuntimeError('Kernel matrix of noise must be diagonal.')

        # And construct the components for the inducing point approximation.
        L_z = B.cholesky(matrix(K_z))
        A = B.eye_from(K_z) + B.qf(K_n, B.transpose(B.trisolve(L_z, K_zx)))
        y_bar = y - e.mean(x) - self.means[p_x](x)
        prod_y_bar = B.trisolve(L_z, B.qf(K_n, B.transpose(K_zx), y_bar))

        return K_z, L_z, K_zx, K_n, A, y_bar, prod_y_bar

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

    @_dispatch(FunctionType, FunctionType)
    def condition(self, build_posterior_kernel, build_posterior_mean):
        # Store prior if it isn't already.
        if not self.prior_kernels and not self.prior_means:
            self.prior_kernels = self.kernels
            self.prior_means = self.means

        # Update to posterior.
        self.kernels = LazyMatrix()
        self.kernels.add_rule((None, None), self.pids, build_posterior_kernel)
        self.means = LazyVector()
        self.means.add_rule((None,), self.pids, build_posterior_mean)

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
        if self.prior_kernels and self.prior_means:
            self.kernels = self.prior_kernels
            self.means = self.prior_means

    def checkpoint(self):
        """Get a checkpoint of the current state of the graph, so that it can
        be reverted to at a later point, say after conditioning.

        Returns:
            tuple: Checkpoint.
        """
        return self.kernels, self.means

    def revert(self, checkpoint):
        """Revert to a checkpoint.

        Args:
            checkpoint (tuple): Checkpoint to revert to.
        """
        kernels, means = checkpoint
        self.kernels = kernels
        self.means = means

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
    def __init__(self, kernel, mean=None, graph=model, name=None):
        # First resolve `kernel` and `mean` through `GPPrimitive`s constructor.
        GPPrimitive.__init__(self, kernel, mean)

        # Then add a new `GP` to the graph with the resolved kernel and mean.
        self.graph = graph
        self.graph.add_independent_gp(self,
                                      GPPrimitive.kernel.fget(self),
                                      GPPrimitive.mean.fget(self))

        # If a name is given, set the name.
        if name:
            self.graph.name(self, name)

    @_dispatch(Graph)
    def __init__(self, graph):
        GPPrimitive.__init__(self, None)
        self.graph = graph

    @property
    def kernel(self):
        """Kernel of the GP."""
        return self.graph.kernels[self]

    @property
    def mean(self):
        """Mean function of the GP."""
        return self.graph.means[self]

    @property
    def name(self):
        """Name of the GP."""
        return self.graph[self]

    @name.setter
    @_dispatch(str)
    def name(self, name):
        self.graph.name(self, name)

    @_dispatch(object)
    def __add__(self, other):
        return self.graph.sum(self, other)

    @_dispatch(Random)
    def __add__(self, other):
        raise NotImplementedError('Cannot add a GP and a {}.'
                                  ''.format(type(other).__name__))

    @_dispatch(Self)
    def __add__(self, other):
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
        # Careful with the closure!
        self_mean, other_mean = self.graph.means[self], self.graph.means[other]
        return (lambda x, B: self_mean(x, B)) * other + \
               self * (lambda x, B: other_mean(x, B)) + \
               GP(kernel=self.graph.kernels[self] *
                         self.graph.kernels[other] +
                         self.graph.kernels[self, other] *
                         self.graph.kernels[other, self],
                  mean=-self.graph.means[self] *
                       self.graph.means[other],
                  graph=self.graph)

    def _ensure_at(self, x):
        return x if isinstance(x, At) else At(self)(x)

    @_dispatch({B.Numeric, Input}, B.Numeric)
    def condition(self, x, y):
        """Condition the GP. See :meth:`.graph.Graph.condition`."""
        self.graph.condition(self._ensure_at(x), y)
        return self

    @_dispatch([{tuple, list}])
    def condition(self, *pairs):
        # Set any unspecified locations to this process.
        self.graph.condition(*[(self._ensure_at(x), y) for x, y in pairs])
        return self

    @_dispatch({B.Numeric, Input}, B.Numeric, {B.Numeric, Input}, Self)
    def condition(self, x, y, z, e):
        """Sparsely condition the GP. See :meth:`.graph.Graph.condition`."""
        self.graph.condition(self._ensure_at(x), y, self._ensure_at(z), e)
        return self

    @_dispatch({B.Numeric, Input}, B.Numeric, {B.Numeric, Input}, Self)
    def elbo(self, x, y, z, e):
        """Compute the ELBO. See :meth:`.graph.Graph.elbo`."""
        return self.graph.elbo(self._ensure_at(x), y, self._ensure_at(z), e)

    @_dispatch(tuple)
    def __or__(self, args):
        return self.condition(*args)

    def __matmul__(self, other):
        """Alternative to writing `At(self)(other)`."""
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
