# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from types import FunctionType

from fdm import central_fdm
from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter, Union

from .util import uprank
from .input import Input, At, MultiInput
from .kernel import ZeroKernel, PosteriorKernel, TensorProductKernel, \
    CorrectiveKernel, OneKernel
from .lazy import LazyVector, LazyMatrix
from .matrix import matrix, Diagonal, dense
from .mean import PosteriorMean, ZeroMean, OneMean
from .mokernel import MultiOutputKernel as MOK
from .momean import MultiOutputMean as MOM
from .random import Random, PromisedGP, RandomProcess, Normal

__all__ = ['GP',
           'model',
           'Graph',
           'AbstractObservations',
           'Observations', 'Obs',
           'SparseObservations', 'SparseObs']

log = logging.getLogger(__name__)


def ensure_at(x, ref=None):
    """Ensure that an input location is typed with `At` to specify which
    process it belongs to.

    Args:
        x (input): Input location.
        ref (:class:`.graph.GP`, optional): Reference process. If provided and
            `x` is not an instance of `At`, then it assumed to belong to `ref`.

    Returns:
        :class:`.input.At`: Input, instance of `At`.
    """
    if isinstance(x, At):
        return x
    elif ref is not None:
        return ref(x)
    else:
        raise ValueError('Must explicitly specify the processes which to '
                         'condition on.')


class AbstractObservations(Referentiable):
    """Abstract base class for observations."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({B.Numeric, Input}, B.Numeric, [PromisedGP])
    def __init__(self, x, y, ref=None):
        self._ref = ref
        self.x = ensure_at(x, self._ref)
        self.y = y
        self.graph = type_parameter(self.x).graph

    @_dispatch([Union(tuple, list, PromisedGP)])
    def __init__(self, *pairs, **kw_args):
        # Check whether there's a reference.
        self._ref = kw_args['ref'] if 'ref' in kw_args else None

        # Ensure `At` for all pairs.
        pairs = [(ensure_at(x, self._ref), y) for x, y in pairs]

        # Get the graph from the first pair.
        self.graph = type_parameter(pairs[0][0]).graph

        # Extend the graph by the Cartesian product `p` of all processes.
        p = self.graph.cross(*self.graph.ps)

        # Condition on the newly created vector-valued GP.
        xs, ys = zip(*pairs)
        self.x = p(MultiInput(*xs))
        self.y = B.concat(*[uprank(y) for y in ys], axis=0)

    @_dispatch({tuple, list})
    def __ror__(self, ps):
        return self.graph.condition(ps, self)

    def posterior_kernel(self, p_i, p_j):  # pragma: no cover
        """Get the posterior kernel between two processes.

        Args:
            p_i (:class:`.graph.GP`): First process.
            p_j (:class:`.graph.GP`): Second process.

        Returns:
            :class:`.kernel.Kernel`: Posterior kernel between the first and
                second process.
        """
        raise NotImplementedError('Posterior kernel construction not '
                                  'implemented.')

    def posterior_mean(self, p):  # pragma: no cover
        """Get the posterior kernel of a process.

        Args:
            p (:class:`.graph.GP`): Process.

        Returns:
            :class:`.mean.Mean`: Posterior mean of `p`.
        """
        raise NotImplementedError('Posterior mean construction not '
                                  'implemented.')


class Observations(AbstractObservations, Referentiable):
    """Observations.

    Can alternatively construct an instance of `Observations` with tuples or
    lists of valid constructors.

    Args:
        x (input): Locations of points to condition on.
        y (tensor): Observations to condition on.
        ref (:class:`.class.GP`, optional): Reference process. See
            :func:`.graph.ensure_at`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, *args, **kw_args):
        AbstractObservations.__init__(self, *args, **kw_args)
        self._K_x = None

    @property
    def K_x(self):
        """Kernel matrix of the data."""
        if self._K_x is None:  # Cache computation.
            p_x, x = type_parameter(self.x), self.x.get()
            self._K_x = matrix(self.graph.kernels[p_x](x))
        return self._K_x

    def posterior_kernel(self, p_i, p_j):
        p_x, x = type_parameter(self.x), self.x.get()
        return PosteriorKernel(self.graph.kernels[p_i, p_j],
                               self.graph.kernels[p_x, p_i],
                               self.graph.kernels[p_x, p_j],
                               x, self.K_x)

    def posterior_mean(self, p):
        p_x, x = type_parameter(self.x), self.x.get()
        return PosteriorMean(self.graph.means[p],
                             self.graph.means[p_x],
                             self.graph.kernels[p_x, p],
                             x, self.K_x, self.y)


class SparseObservations(AbstractObservations, Referentiable):
    """Observations through inducing points. Takes further arguments
    according to the constructor of :class:`.graph.Observations`.

    Attributes:
        elbo (scalar): ELBO.

    Args:
        z (input): Locations of the inducing points.
        e (:class:`.graph.GP`): Additive, independent noise process.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch({B.Numeric, Input, tuple, list},
               [Union(tuple, list, PromisedGP)])
    def __init__(self, z, *pairs, **kw_args):
        es, xs, ys = zip(*pairs)
        AbstractObservations.__init__(self, *zip(xs, ys), **kw_args)
        SparseObservations.__init__(self,
                                    z,
                                    self.graph.cross(*es),
                                    self.x,
                                    self.y,
                                    **kw_args)

    @_dispatch({list, tuple},
               PromisedGP,
               {B.Numeric, Input},
               B.Numeric,
               [PromisedGP])
    def __init__(self, zs, e, x, y, ref=None):
        # Ensure `At` everywhere.
        zs = [ensure_at(z, ref=ref) for z in zs]

        # Extract graph.
        graph = type_parameter(zs[0]).graph

        # Create a representative multi-output process.
        p_z = graph.cross(*(type_parameter(z) for z in zs))

        SparseObservations.__init__(self,
                                    p_z(MultiInput(*zs)),
                                    e, x, y, ref=ref)

    @_dispatch({B.Numeric, Input},
               PromisedGP,
               {B.Numeric, Input},
               B.Numeric,
               [PromisedGP])
    def __init__(self, z, e, x, y, ref=None):
        AbstractObservations.__init__(self, x, y, ref=ref)
        self.z = ensure_at(z, self._ref)
        self.e = e

        self._K_z = None
        self._elbo = None
        self._mu = None
        self._A = None

    @property
    def K_z(self):
        """Kernel matrix of the data."""
        if self._K_z is None:  # Cache computation.
            self._compute()
        return self._K_z

    @property
    def elbo(self):
        """ELBO."""
        if self._elbo is None:  # Cache computation.
            self._compute()
        return self._elbo

    @property
    def mu(self):
        """Mean of optimal approximating distribution."""
        if self._mu is None:  # Cache computation.
            self._compute()
        return self._mu

    @property
    def A(self):
        """Parameter of the corrective variance of the kernel of the optimal
        approximating distribution."""
        if self._A is None:  # Cache computation.
            self._compute()
        return self._A

    def _compute(self):
        # Extract processes.
        p_x, x = type_parameter(self.x), self.x.get()
        p_z, z = type_parameter(self.z), self.z.get()

        # Construct the necessary kernel matrices.
        K_zx = self.graph.kernels[p_z, p_x](z, x)
        self._K_z = matrix(self.graph.kernels[p_z](z))

        # Evaluating `e.kernel(x)` will yield incorrect results if `x` is a
        # `MultiInput`, because `x` then still designates the particular
        # components of `f`. Fix that by instead designating the elements of
        # `e`.
        if isinstance(x, MultiInput):
            x_n = MultiInput(*(p(xi.get())
                               for p, xi in zip(self.e.kernel.ps, x.get())))
        else:
            x_n = x

        # Construct the noise kernel matrix.
        K_n = self.e.kernel(x_n)

        # The approximation can only handle diagonal noise matrices.
        if not isinstance(K_n, Diagonal):
            raise RuntimeError('Kernel matrix of noise must be diagonal.')

        # And construct the components for the inducing point approximation.
        L_z = B.cholesky(self._K_z)
        self._A = B.eye(self._K_z) + \
                  B.qf(K_n, B.transpose(B.trisolve(L_z, K_zx)))
        y_bar = uprank(self.y) - self.e.mean(x_n) - self.graph.means[p_x](x)
        prod_y_bar = B.trisolve(L_z, B.qf(K_n, B.transpose(K_zx), y_bar))

        # Compute the optimal mean.
        self._mu = self.graph.means[p_z](z) + \
                   B.qf(self._A, B.trisolve(L_z, self._K_z), prod_y_bar)

        # Compute the ELBO.
        # NOTE: The calculation of `trace_part` asserts that `K_n` is diagonal.
        #       The rest, however, is completely generic.
        trace_part = B.ratio(Diagonal(self.graph.kernels[p_x].elwise(x)[:, 0]) -
                             Diagonal(B.qf_diag(self._K_z, K_zx)), K_n)
        det_part = B.logdet(2 * B.pi * K_n) + B.logdet(self._A)
        qf_part = B.qf(K_n, y_bar)[0, 0] - B.qf(self._A, prod_y_bar)[0, 0]
        self._elbo = -0.5 * (trace_part + det_part + qf_part)

    def posterior_kernel(self, p_i, p_j):
        p_z, z = type_parameter(self.z), self.z.get()
        return PosteriorKernel(self.graph.kernels[p_i, p_j],
                               self.graph.kernels[p_z, p_i],
                               self.graph.kernels[p_z, p_j],
                               z, self.K_z) + \
               CorrectiveKernel(self.graph.kernels[p_z, p_i],
                                self.graph.kernels[p_z, p_j],
                                z, self.A, self.K_z)

    def posterior_mean(self, p):
        p_z, z = type_parameter(self.z), self.z.get()
        return PosteriorMean(self.graph.means[p],
                             self.graph.means[p_z],
                             self.graph.kernels[p_z, p],
                             z, self.K_z, self.mu)


Obs = Observations  #: Shorthand for `Observations`.
SparseObs = SparseObservations  #: Shorthand for `SparseObservations`.


class Graph(Referentiable):
    """A GP model."""
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self.ps = []
        self.pids = set()
        self.kernels = LazyMatrix()
        self.means = LazyVector()

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
        self.kernels.add_rule((None, p), self.pids,
                              lambda pi: reversed(self.kernels[p, pi]))
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
        return self._update(self.means[p] + other,
                            lambda: self.kernels[p],
                            lambda pi: self.kernels[p, pi])

    @_dispatch(PromisedGP, PromisedGP)
    def sum(self, p1, p2):
        # Check that the GPs are on the same graph.
        if p1.graph != p2.graph:
            raise RuntimeError('Can only add GPs from the same graph.')

        return self._update(self.means[p1] + self.means[p2],
                            (lambda: self.kernels[p1] +
                                     self.kernels[p2] +
                                     self.kernels[p1, p2] +
                                     self.kernels[p2, p1]),
                            lambda pi: self.kernels[p1, pi] +
                                       self.kernels[p2, pi])

    @_dispatch(PromisedGP, B.Numeric)
    def mul(self, p, other):
        """Multiply a GP from the graph with another object.

        Args:
            p (:class:`.graph.GP`): GP in the product.
            other (object): Other object in the product.

        Returns:
            :class:`.graph.GP`: The GP corresponding to the product.
        """
        return self._update(self.means[p] * other,
                            lambda: self.kernels[p] * other ** 2,
                            lambda pi: self.kernels[p, pi] * other)

    @_dispatch(PromisedGP, FunctionType)
    def mul(self, p, f):
        def ones(x):
            return B.ones(B.dtype(x), B.shape(x)[0], 1)

        return self._update(f * self.means[p],
                            lambda: f * self.kernels[p],
                            (lambda pi: TensorProductKernel(f, ones) *
                                        self.kernels[p, pi]))

    def shift(self, p, shift):
        """Shift a GP.

        Args:
            p (:class:`.graph.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            :class:`.graph.GP`: The shifted GP.
        """
        return self._update(self.means[p].shift(shift),
                            lambda: self.kernels[p].shift(shift),
                            lambda pi: self.kernels[p, pi].shift(shift, 0))

    def stretch(self, p, stretch):
        """Stretch a GP.

        Args:
            p (:class:`.graph.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            :class:`.graph.GP`: The stretched GP.
        """
        return self._update(self.means[p].stretch(stretch),
                            lambda: self.kernels[p].stretch(stretch),
                            lambda pi: self.kernels[p, pi].stretch(stretch, 1))

    def select(self, p, *dims):
        """Select input dimensions.

        Args:
            p (:class:`.graph.GP`): GP to select input
                dimensions from.
            *dims (object): Dimensions to select.

        Returns:
            :class:`.graph.GP`: GP with the specific input dimensions.
        """
        return self._update(self.means[p].select(dims),
                            lambda: self.kernels[p].select(dims),
                            lambda pi: self.kernels[p, pi].select(dims, None))

    def transform(self, p, f):
        """Transform the inputs of a GP.

        Args:
            p (:class:`.graph.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            :class:`.graph.GP`: Input-transformed GP.
        """
        return self._update(self.means[p].transform(f),
                            lambda: self.kernels[p].transform(f),
                            lambda pi: self.kernels[p, pi].transform(f, None))

    def diff(self, p, dim=0):
        """Differentiate a GP.

        Args:
            p (:class:`.graph.GP`): GP to differentiate.
            dim (int, optional): Dimension of feature which to take the
                derivative with respect to. Defaults to `0`.

        Returns:
            :class:`.graph.GP`: Derivative of GP.
        """
        return self._update(self.means[p].diff(dim),
                            lambda: self.kernels[p].diff(dim),
                            lambda pi: self.kernels[p, pi].diff(dim, None))

    @_dispatch({list, tuple}, AbstractObservations)
    def condition(self, ps, obs):
        """Condition the graph on observations.

        Args:
            ps (list[:class:`.graph.GP`]): Processes to condition.
            obs (:class:`.graph.AbstractObservations`): Observations to
                condition on.

        Returns:
            list[:class:`.graph.GP`]: Posterior processes.
        """

        # A construction like this is necessary to properly close over `p`.
        def build_gens(p):
            def k_ij_generator(pi):
                return obs.posterior_kernel(p, pi)

            def k_ii_generator():
                return obs.posterior_kernel(p, p)

            return k_ii_generator, k_ij_generator

        return [self._update(obs.posterior_mean(p), *build_gens(p)) for p in ps]

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

    @_dispatch(int, [At])
    def sample(self, n, *xs):
        """Sample multiple processes simultaneously.

        Args:
            n (int, optional): Number of samples. Defaults to `1`.
            *xs (:class:`.graph.At`): Locations to sample at.

        Returns:
            tuple: Tuple of samples.
        """
        sample = GP(MOK(*self.ps),
                    MOM(*self.ps),
                    graph=Graph())(MultiInput(*xs)).sample(n)

        # To unpack `x`, just keep `.get()`ing.
        def unpack(x):
            while isinstance(x, Input):
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

    @_dispatch([{list, tuple}])
    def logpdf(self, *pairs):
        xs, ys = zip(*pairs)

        # Check that all processes are specified.
        if not all([isinstance(x, At) for x in xs]):
            raise ValueError('Must explicitly specify the processes which to '
                             'compute the log-pdf for.')

        # Uprank all outputs and concatenate.
        y = B.concat(*[uprank(y) for y in ys], axis=0)

        # Return composite log-pdf.
        return GP(MOK(*self.ps),
                  MOM(*self.ps),
                  graph=Graph())(MultiInput(*xs)).logpdf(y)

    @_dispatch(At, B.Numeric)
    def logpdf(self, x, y):
        return x.logpdf(y)

    @_dispatch(Observations)
    def logpdf(self, obs):
        return obs.x.logpdf(obs.y)

    @_dispatch(SparseObservations)
    def logpdf(self, obs):
        return obs.elbo


model = Graph()  #: A default graph provided for convenience


class GP(RandomProcess, Referentiable):
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
        # Resolve kernel.
        if isinstance(kernel, (B.Numeric, FunctionType)):
            kernel = kernel * OneKernel()

        # Resolve mean.
        if mean is None:
            mean = ZeroMean()
        elif isinstance(mean, (B.Numeric, FunctionType)):
            mean = mean * OneMean()

        # Then add a new `GP` to the graph with the resolved kernel and mean.
        self.graph = graph
        self.graph.add_independent_gp(self, kernel, mean)

        # If a name is given, set the name.
        if name:
            self.graph.name(self, name)

    @_dispatch(Graph)
    def __init__(self, graph):
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

    def __call__(self, x):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
            x (input): Points to construct the distribution at.

        Returns:
            :class:`.random.Normal`: Finite-dimensional distribution.
        """
        return Normal(self, x)

    @_dispatch([object])
    def condition(self, *args):
        """Condition the GP. See :meth:`.graph.Graph.condition`."""
        return self.graph.condition((self,), Observations(*args, ref=self))[0]

    @_dispatch(AbstractObservations)
    def condition(self, obs):
        return self.graph.condition((self,), obs)[0]

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
        return (lambda x: self.graph.means[self](x)) * other + \
               self * (lambda x: self.graph.means[other](x)) + \
               GP(kernel=self.graph.kernels[self] *
                         self.graph.kernels[other] +
                         self.graph.kernels[self, other] *
                         self.graph.kernels[other, self],
                  mean=-self.graph.means[self] *
                       self.graph.means[other],
                  graph=self.graph)

    @_dispatch([object])
    def __or__(self, args):
        """Shorthand for conditioning."""
        return self.condition(Observations(*args, ref=self))

    @_dispatch(AbstractObservations)
    def __or__(self, obs):
        return self.condition(obs)

    def shift(self, shift):
        """Shift the GP. See :meth:`.graph.Graph.shift`."""
        return self.graph.shift(self, shift)

    def stretch(self, stretch):
        """Stretch the GP. See :meth:`.graph.Graph.stretch`."""
        return self.graph.stretch(self, stretch)

    def __gt__(self, stretch):
        """Shorthand for :meth:`.graph.GP.stretch`."""
        return self.stretch(stretch)

    def transform(self, f):
        """Input transform the GP. See :meth:`.graph.Graph.transform`."""
        return self.graph.transform(self, f)

    def select(self, *dims):
        """Select dimensions from the input. See :meth:`.graph.Graph.select`."""
        return self.graph.select(self, *dims)

    def __getitem__(self, *dims):
        """Shorthand for :meth:`.graph.GP.select`."""
        return self.select(*dims)

    def diff(self, dim=0):
        """Differentiate the GP. See :meth:`.graph.Graph.diff`."""
        return self.graph.diff(self, dim)

    def diff_approx(self, deriv=1, order=6):
        """Approximate the derivative of the GP by constructing a finite
        difference approximation.

        Args:
            deriv (int): Order of the derivative.
            order (int): Order of the estimate.

        Returns:
            Approximation of the derivative of the GP.
        """
        # Use the FDM library to figure out the coefficients.
        fdm = central_fdm(order, deriv, adapt=0, factor=1e8)
        fdm.estimate()  # Estimate step size.

        # Construct finite difference.
        df = 0
        for g, c in zip(fdm.grid, fdm.coefs):
            df += c * self.shift(-g * fdm.step)
        return df / fdm.step ** deriv

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


PromisedGP.deliver(GP)
