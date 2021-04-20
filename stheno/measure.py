import logging
from types import FunctionType

from fdm import central_fdm
from lab import B
from matrix import Diagonal, AbstractMatrix, Constant
from plum import Dispatcher, Union, convert

from . import PromisedFDD, PromisedGP
from .input import MultiInput
from .kernel import (
    ZeroKernel,
    PosteriorKernel,
    TensorProductKernel,
    CorrectiveKernel,
    OneKernel,
)
from .lazy import LazyVector, LazyMatrix
from .mean import PosteriorMean, ZeroMean, OneMean
from .mokernel import MultiOutputKernel as MOK
from .momean import MultiOutputMean as MOM
from .random import Random, RandomProcess, Normal
from .util import uprank, num_elements

__all__ = [
    "GP",
    "cross",
    "Measure",
    "AbstractObservations",
    "Observations",
    "Obs",
    "SparseObservations",
    "SparseObs",
]

_dispatch = Dispatcher()

log = logging.getLogger(__name__)


def assert_same_measure(*ps):
    """Assert that processes are associated to the same measure.

    Args:
        *ps (:class:`.measure.GP`): Processes.
    """
    # First check that all processes are constructed from the same measure.
    for p in ps[1:]:
        if ps[0].measure != p.measure:
            raise AssertionError(
                f"Processes {ps[0]} and {p} are associated to different measures."
            )


def intersection_measure_group(*ps):
    """Assert that processes have equal measure groups.

    Args:
        *ps (:class:`.measure.GP`): Processes.
    """
    assert_same_measure(*ps)
    intersection = set(ps[0]._measures)
    for p in ps[1:]:
        intersection &= set(p._measures)
    return intersection


class AbstractObservations:
    """Abstract base class for observations.

    Attributes:
        fdd (:class:`.measure.FDD`): FDD of observations.
        y (column vector): Values of observations.
    """

    @_dispatch
    def __init__(self, fdd: Union[PromisedFDD], y: Union[B.Numeric, AbstractMatrix]):
        self.fdd = fdd
        self.y = y

    @_dispatch
    def __init__(self, *pairs: tuple):
        fdds, ys = zip(*pairs)
        self.fdd = cross(*[fdd.p for fdd in fdds])(MultiInput(*fdds))
        self.y = B.concat(*[uprank(y) for y in ys], axis=0)

    def posterior_kernel(self, measure, p_i, p_j):  # pragma: no cover
        """Get the posterior kernel between two processes.

        Args:
            measure (:class:`.measure.Measure`): Prior.
            p_i (:class:`.measure.GP`): First process.
            p_j (:class:`.measure.GP`): Second process.

        Returns:
            :class:`.kernel.Kernel`: Posterior kernel between the first and
                second process.
        """
        raise NotImplementedError("Posterior kernel construction not implemented.")

    def posterior_mean(self, measure, p):  # pragma: no cover
        """Get the posterior kernel of a process.

        Args:
            measure (:class:`.measure.Measure`): Prior.
            p (:class:`.measure.GP`): Process.

        Returns:
            :class:`.mean.Mean`: Posterior mean of `p`.
        """
        raise NotImplementedError("Posterior mean construction not implemented.")


class Observations(AbstractObservations):
    """Observations.

    Can alternatively construct an instance of `Observations` with tuples of valid
    constructors.

    Args:
        fdd (:class:`.measure.FDD`): FDDs to corresponding to the observations.
        y (tensor): Values of observations.
    """

    def __init__(self, *args):
        AbstractObservations.__init__(self, *args)
        self._K_x_store = {}

    def K_x(self, measure):
        """Kernel matrix of the data.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Kernel matrix.
        """
        try:
            return self._K_x_store[id(measure)]
        except KeyError:
            K_x = measure.kernels[self.fdd.p](self.fdd.x)
            self._K_x_store[id(measure)] = K_x
            return K_x

    def posterior_kernel(self, measure, p_i, p_j):
        if num_elements(self.fdd.x) == 0:
            # There are no observations! Just return prior.
            return measure.kernels[p_i, p_j]

        return PosteriorKernel(
            measure.kernels[p_i, p_j],
            measure.kernels[self.fdd.p, p_i],
            measure.kernels[self.fdd.p, p_j],
            self.fdd.x,
            self.K_x(measure),
        )

    def posterior_mean(self, measure, p):
        if num_elements(self.fdd.x) == 0:
            # There are no observations! Just return prior.
            return measure.means[p]

        return PosteriorMean(
            measure.means[p],
            measure.means[self.fdd.p],
            measure.kernels[self.fdd.p, p],
            self.fdd.x,
            self.K_x(measure),
            self.y,
        )


class SparseObservations(AbstractObservations):
    """Observations through inducing points.

    Takes further arguments according to the constructor of
    :class:`.measure.Observations`.

    Args:
        u (:class:`.measure.FDD`): Inducing points
        e (:class:`.measure.GP`): Additive, independent noise process.
    """

    @_dispatch
    def __init__(self, u: Union[tuple, PromisedFDD], *pairs: tuple):
        es, fdds, ys = zip(*pairs)

        # Copy the noises to a measure under which they are independent.
        measure = Measure()
        e = cross(*[GP(e.mean, e.kernel, measure=measure) for e in es])

        fdd = cross(*[fdd.p for fdd in fdds])(MultiInput(*fdds))
        y = B.concat(*[uprank(y) for y in ys], axis=0)
        SparseObservations.__init__(self, u, e, fdd, y)

    @_dispatch
    def __init__(self, us: tuple, e: PromisedGP, fdd: PromisedFDD, y: B.Numeric):
        u = cross(*[u.p for u in us])(MultiInput(*us))
        SparseObservations.__init__(self, u, e, fdd, y)

    @_dispatch
    def __init__(self, u: PromisedFDD, e: PromisedGP, fdd: PromisedFDD, y: B.Numeric):
        AbstractObservations.__init__(self, fdd, y)
        self.u = u
        self.e = e
        self._K_z_store = {}
        self._elbo_store = {}
        self._mu_store = {}
        self._A_store = {}

    def K_z(self, measure):
        """Kernel matrix of the data.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Kernel matrix.
        """
        try:
            return self._K_z_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._K_z_store[id(measure)]

    def elbo(self, measure):
        """ELBO.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            scalar: ELBO.
        """
        try:
            return self._elbo_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._elbo_store[id(measure)]

    def mu(self, measure):
        """Mean of optimal approximating distribution.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Mean.
        """
        try:
            return self._mu_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._mu_store[id(measure)]

    def A(self, measure):
        """Parameter of the corrective variance of the kernel of the optimal
        approximating distribution.

        Args:
            measure (:class:`.measure.Measure`): Measure.

        Returns:
            matrix: Corrective variance.
        """
        try:
            return self._A_store[id(measure)]
        except KeyError:
            self._compute(measure)
            return self._A_store[id(measure)]

    def _compute(self, measure):
        # Extract processes and inputs.
        p_x, x = self.fdd.p, self.fdd.x
        p_z, z = self.u.p, self.u.x

        # Construct the necessary kernel matrices.
        K_zx = measure.kernels[p_z, p_x](z, x)
        K_z = convert(measure.kernels[p_z](z), AbstractMatrix)
        self._K_z_store[id(measure)] = K_z

        # Evaluating `e.kernel(x)` will yield incorrect results if `x` is a
        # `MultiInput`, because `x` then still designates the particular components
        # of `f`. Fix that by instead designating the elements of `e`.
        if isinstance(x, MultiInput):
            x_n = MultiInput(*(e(fdd.x) for e, fdd in zip(self.e.kernel.ps, x.get())))
        else:
            x_n = x

        # Construct the noise kernel matrix.
        K_n = self.e.kernel(x_n)

        # The approximation can only handle diagonal noise matrices.
        if not isinstance(K_n, Diagonal):
            raise RuntimeError("Kernel matrix of noise must be diagonal.")

        # And construct the components for the inducing point approximation.
        L_z = B.cholesky(K_z)
        A = B.add(B.eye(K_z), B.iqf(K_n, B.transpose(B.solve(L_z, K_zx))))
        self._A_store[id(measure)] = A
        y_bar = uprank(self.y) - self.e.mean(x_n) - measure.means[p_x](x)
        prod_y_bar = B.solve(L_z, B.iqf(K_n, B.transpose(K_zx), y_bar))

        # Compute the optimal mean.
        mu = B.add(
            measure.means[p_z](z),
            B.iqf(A, B.solve(L_z, K_z), prod_y_bar),
        )
        self._mu_store[id(measure)] = mu

        # Compute the ELBO.
        # NOTE: The calculation of `trace_part` asserts that `K_n` is diagonal.
        # The rest, however, is completely generic.
        trace_part = B.ratio(
            Diagonal(measure.kernels[p_x].elwise(x)[:, 0])
            - Diagonal(B.iqf_diag(K_z, K_zx)),
            K_n,
        )
        det_part = B.logdet(2 * B.pi * K_n) + B.logdet(A)
        iqf_part = B.iqf(K_n, y_bar)[0, 0] - B.iqf(A, prod_y_bar)[0, 0]
        self._elbo_store[id(measure)] = -0.5 * (trace_part + det_part + iqf_part)

    def posterior_kernel(self, measure, p_i, p_j):
        return PosteriorKernel(
            measure.kernels[p_i, p_j],
            measure.kernels[self.u.p, p_i],
            measure.kernels[self.u.p, p_j],
            self.u.x,
            self.K_z(measure),
        ) + CorrectiveKernel(
            measure.kernels[self.u.p, p_i],
            measure.kernels[self.u.p, p_j],
            self.u.x,
            self.A(measure),
            self.K_z(measure),
        )

    def posterior_mean(self, measure, p):
        return PosteriorMean(
            measure.means[p],
            measure.means[self.u.p],
            measure.kernels[self.u.p, p],
            self.u.x,
            self.K_z(measure),
            self.mu(measure),
        )


Obs = Observations  #: Shorthand for `Observations`.
SparseObs = SparseObservations  #: Shorthand for `SparseObservations`.


class Measure:
    """A GP model.

    Attributes:
        ps (list[:class:`.measure.GP`]): Processes of the measure.
        mean (:class:`.lazy.LazyVector`): Mean.
        kernels (:class:`.lazy.LazyMatrix`): Kernels.
    """

    def __init__(self):
        self.ps = []
        self._pids = set()
        self.means = LazyVector()
        self.kernels = LazyMatrix()

        # Store named GPs in both ways.
        self._gps_by_name = {}
        self._names_by_gp = {}

    def __hash__(self):
        # This is needed for :func:`.measure.intersection_measure_group`, which puts
        # many :class:`.measure.Measure`s in a `set`. Every measure is unique.
        return id(self)

    @_dispatch
    def __getitem__(self, name: str):
        return self._gps_by_name[name]

    @_dispatch
    def __getitem__(self, p: PromisedGP):
        return self._names_by_gp[id(p)]

    @_dispatch
    def name(self, p: PromisedGP, name: str):
        """Name a GP.

        Args:
            p (:class:`.measure.GP`): GP to name.
            name (str): Name. Must be unique.
        """
        # Delete any existing names and back-references for the GP.
        if id(p) in self._names_by_gp:
            del self._gps_by_name[self._names_by_gp[id(p)]]
            del self._names_by_gp[id(p)]

        # Check that name is not in use.
        if name in self._gps_by_name:
            raise RuntimeError(
                f'Name "{name}" for "{p}" already taken by "{self[name]}".'
            )

        # Set the name and the back-reference.
        self._gps_by_name[name] = p
        self._names_by_gp[id(p)] = name

    def _add_p(self, p):
        # Attach process to measure.
        self.ps.append(p)
        self._pids.add(id(p))
        # Add measure to list of measures of process.
        p._measures.append(self)

    def _update(self, p, mean, kernel, left_rule, right_rule=None):
        # Update means.
        self.means[p] = mean

        # Update kernels.
        self.kernels[p] = kernel
        self.kernels.add_left_rule(id(p), self._pids, left_rule)
        if right_rule:
            self.kernels.add_right_rule(id(p), self._pids, right_rule)
        else:
            self.kernels.add_right_rule(
                id(p), self._pids, lambda i: reversed(self.kernels[p, i])
            )

        # Only now add `p`: `self.pids` above needs to not include `id(p)`.
        self._add_p(p)

        return p

    @_dispatch
    def __call__(self, p: PromisedGP):
        # Make a new GP with `self` as the prior.
        p_copy = GP()
        return self._update(
            p_copy,
            self.means[p],
            self.kernels[p],
            # `p_copy` acts like `p`.
            lambda j: self.kernels[p, j],  # Left rule
            lambda i: self.kernels[i, p],  # Right rule
        )

    @_dispatch
    def __call__(self, fdd: PromisedFDD):
        return Normal(self.means[fdd.p](fdd.x), self.kernels[fdd.p](fdd.x))

    def add_independent_gp(self, p, mean, kernel):
        """Add an independent GP to the model.

        Args:
            p (:class:`.measure.GP`): GP to add.
            mean (:class:`.mean.Mean`): Mean function of GP.
            kernel (:class:`.kernel.Kernel`): Kernel function of GP.

        Returns:
            :class:`.measure.GP`: The newly added independent GP.
        """
        # Update means.
        self.means[p] = mean

        # Update kernels.
        self.kernels[p] = kernel
        self.kernels.add_left_rule(id(p), self._pids, lambda j: ZeroKernel())
        self.kernels.add_right_rule(id(p), self._pids, lambda i: ZeroKernel())

        # Only now add `p`: `self.pids` above needs to not include `id(p)`.
        self._add_p(p)

        return p

    @_dispatch
    def sum(self, p_sum: PromisedGP, other, p: PromisedGP):
        """Sum a GP from the graph with another object.

        Args:
            p_sum (:class:`.measure.GP`): GP that is the sum.
            obj1 (other type or :class:`.measure.GP`): First term in the sum.
            obj2 (other type or :class:`.measure.GP`): Second term in the sum.

        Returns:
            :class:`.measure.GP`: The GP corresponding to the sum.
        """
        return self.sum(p_sum, p, other)

    @_dispatch
    def sum(
        self, p_sum: PromisedGP, p: PromisedGP, other: Union[B.Numeric, FunctionType]
    ):
        return self._update(
            p_sum,
            self.means[p] + other,
            self.kernels[p],
            lambda j: self.kernels[p, j],
        )

    @_dispatch
    def sum(self, p_sum: PromisedGP, p1: PromisedGP, p2: PromisedGP):
        assert_same_measure(p1, p2)
        return self._update(
            p_sum,
            self.means[p1] + self.means[p2],
            (
                self.kernels[p1]
                + self.kernels[p2]
                + self.kernels[p1, p2]
                + self.kernels[p2, p1]
            ),
            lambda j: self.kernels[p1, j] + self.kernels[p2, j],
        )

    @_dispatch
    def mul(self, p_mul: PromisedGP, other, p: PromisedGP):
        """Multiply a GP from the graph with another object.

        Args:
            p_mul (:class:`.measure.GP`): GP that is the product.
            obj1 (object): First factor in the product.
            obj2 (object): Second factor in the product.
            other (object): Other object in the product.

        Returns:
            :class:`.measure.GP`: The GP corresponding to the product.
        """
        return self.mul(p_mul, p, other)

    @_dispatch
    def mul(self, p_mul: PromisedGP, p: PromisedGP, other: B.Numeric):
        return self._update(
            p_mul,
            self.means[p] * other,
            self.kernels[p] * other ** 2,
            lambda j: self.kernels[p, j] * other,
        )

    @_dispatch
    def mul(self, p_mul: PromisedGP, p: PromisedGP, f: FunctionType):
        def ones(x):
            return Constant(B.one(x), num_elements(x), 1)

        return self._update(
            p_mul,
            f * self.means[p],
            f * self.kernels[p],
            lambda j: TensorProductKernel(f, ones) * self.kernels[p, j],
        )

    @_dispatch
    def mul(self, p_mul: PromisedGP, p1: PromisedGP, p2: PromisedGP):
        assert_same_measure(p1, p2)
        term1 = self.sum(
            GP(),
            self.mul(GP(), lambda x: self.means[p1](x), p2),
            self.mul(GP(), p1, lambda x: self.means[p2](x)),
        )
        term2 = self.add_independent_gp(
            GP(),
            -self.means[p1] * self.means[p2],
            (
                self.kernels[p1] * self.kernels[p2]
                + self.kernels[p1, p2] * self.kernels[p2, p1]
            ),
        )
        return self.sum(p_mul, term1, term2)

    def shift(self, p_shifted, p, shift):
        """Shift a GP.

        Args:
            p_shifted (:class:`.measure.GP`): Shifted GP.
            p (:class:`.measure.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            :class:`.measure.GP`: The shifted GP.
        """
        return self._update(
            p_shifted,
            self.means[p].shift(shift),
            self.kernels[p].shift(shift),
            lambda j: self.kernels[p, j].shift(shift, 0),
        )

    def stretch(self, p_stretched, p, stretch):
        """Stretch a GP.

        Args:
            p_stretched (:class:`.measure.GP`): Stretched GP.
            p (:class:`.measure.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            :class:`.measure.GP`: The stretched GP.
        """
        return self._update(
            p_stretched,
            self.means[p].stretch(stretch),
            self.kernels[p].stretch(stretch),
            lambda j: self.kernels[p, j].stretch(stretch, 1),
        )

    def select(self, p_selected, p, *dims):
        """Select input dimensions.

        Args:
            p_selected (:class:`.measure.GP`): GP with selected inputs.
            p (:class:`.measure.GP`): GP to select input dimensions from.
            *dims (object): Dimensions to select.

        Returns:
            :class:`.measure.GP`: GP with the specific input dimensions.
        """
        return self._update(
            p_selected,
            self.means[p].select(dims),
            self.kernels[p].select(dims),
            lambda j: self.kernels[p, j].select(dims, None),
        )

    def transform(self, p_transformed, p, f):
        """Transform the inputs of a GP.

        Args:
            p_transformed (:class:`.measure.GP`): GP with transformed inputs.
            p (:class:`.measure.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            :class:`.measure.GP`: Input-transformed GP.
        """
        return self._update(
            p_transformed,
            self.means[p].transform(f),
            self.kernels[p].transform(f),
            lambda j: self.kernels[p, j].transform(f, None),
        )

    def diff(self, p_diff, p, dim=0):
        """Differentiate a GP.

        Args:
            p_diff (:class:`.measure.GP`): Derivative.
            p (:class:`.measure.GP`): GP to differentiate.
            dim (int, optional): Dimension of feature which to take the derivative
                with respect to. Defaults to `0`.

        Returns:
            :class:`.measure.GP`: Derivative of GP.
        """
        return self._update(
            p_diff,
            self.means[p].diff(dim),
            self.kernels[p].diff(dim),
            lambda j: self.kernels[p, j].diff(dim, None),
        )

    @_dispatch
    def condition(self, obs: AbstractObservations):
        """Condition the measure on observations.

        Args:
            obs (:class:`.measure.AbstractObservations`): Observations to condition on.

        Returns:
            list[:class:`.measure.GP`]: Posterior processes.
        """
        posterior = Measure()
        posterior.ps = list(self.ps)
        posterior._pids = set(self._pids)

        posterior.means.add_rule(posterior._pids, lambda i: obs.posterior_mean(self, i))
        posterior.kernels.add_rule(
            posterior._pids, lambda i, j: obs.posterior_kernel(self, i, j)
        )

        # Update backreferences.
        for p in posterior.ps:
            p._measures.append(posterior)

        return posterior

    @_dispatch
    def condition(self, fdd: PromisedFDD, y: B.Numeric):
        return self.condition(Obs(fdd, y))

    @_dispatch
    def condition(self, pair: tuple):
        return self.condition(Obs(*pair))

    @_dispatch
    def condition(self, *pairs: tuple):
        return self.condition(Obs(*pairs))

    @_dispatch
    def __or__(self, *args):
        return self.condition(*args)

    @_dispatch
    def cross(self, p_cross: PromisedGP, *ps: PromisedGP):
        """Construct the Cartesian product of a collection of processes.

        Args:
            p_cross (:class:`.measure.GP`): GP that is the Cartesian product.
            *ps (:class:`.measure.GP`): Processes to construct the Cartesian product of.

        Returns:
            :class:`.measure.GP`: The Cartesian product of `ps`.
        """
        mok = MOK(self, *ps)
        return self._update(
            p_cross,
            MOM(self, *ps),
            mok,
            lambda j: mok.transform(None, lambda y: FDD(j, y)),
        )

    @_dispatch
    def sample(self, n: int, *fdds: PromisedFDD):
        """Sample multiple processes simultaneously.

        Args:
            n (int, optional): Number of samples. Defaults to `1`.
            *fdds (:class:`.measure.FDD`): Locations to sample at.

        Returns:
            tuple: Tuple of samples.
        """
        sample = cross(*self.ps)(MultiInput(*fdds)).sample(n)

        # Unpack sample.
        lengths = [num_elements(fdd) for fdd in fdds]
        i, samples = 0, []
        for length in lengths:
            samples.append(sample[i : i + length, :])
            i += length
        return samples[0] if len(samples) == 1 else samples

    @_dispatch
    def sample(self, *fdds: PromisedFDD):
        return self.sample(1, *fdds)

    @_dispatch
    def logpdf(self, *pairs: Union[list, tuple]):
        """Compute the logpdf of one multiple observations.

        Can also give an `AbstractObservations`.

        Args:
            *pairs (tuple[:class:`.measure.FDD`, tensor]): Pairs of FDDs and values
                of the observations.

        Returns:
            scalar: Logpdf.
        """
        fdds, ys = zip(*pairs)
        y = B.concat(*[uprank(y) for y in ys], axis=0)
        return self(
            self.cross(GP(), *[fdd.p for fdd in fdds])(MultiInput(*fdds))
        ).logpdf(y)

    @_dispatch
    def logpdf(self, fdd: PromisedFDD, y: B.Numeric):
        return self(fdd).logpdf(y)

    @_dispatch
    def logpdf(self, obs: Observations):
        return self.logpdf(obs.fdd, obs.y)

    @_dispatch
    def logpdf(self, obs: SparseObservations):
        return obs.elbo(self)


class GP(RandomProcess):
    """Gaussian process.

    Args:
        mean (:class:`.mean.Mean`, optional): Mean function of the process. Defaults
            to zero.
        kernel (:class:`.kernel.Kernel`): Kernel of the process.
        measure (:class:`.measure.Measure`): Measure to attach to. Must be given as
            a keyword argument.
        name (:obj:`str`, optional): Name. Must be given as a keyword argment.
    """

    @_dispatch
    def __init__(self, mean, kernel, measure=None, name=None):
        self._measures = []

        # If no measure is given, create one.
        if measure is None:
            measure = Measure()

        # Resolve mean.
        if isinstance(mean, (B.Numeric, FunctionType)):
            mean = mean * OneMean()

        # Resolve kernel.
        if isinstance(kernel, (B.Numeric, FunctionType)):
            kernel = kernel * OneKernel()

        # Add a new `GP` to the measure with the resolved kernel and mean. The measure
        # will add itself to `self.measures`.
        measure.add_independent_gp(self, mean, kernel)

        # If a name is given, set the name.
        if name:
            measure.name(self, name)

    @_dispatch
    def __init__(self, kernel, measure=None, name=None):
        self.__init__(ZeroMean(), kernel, measure=measure, name=name)

    @_dispatch
    def __init__(self):
        self._measures = []

    @property
    def measure(self):
        """Measure that the GP was constructed with."""
        if len(self._measures) == 0:
            raise RuntimeError("GP is not associated to a measure.")
        else:
            return self._measures[0]

    @property
    def kernel(self):
        """Kernel of the GP."""
        return self.measure.kernels[self]

    @property
    def mean(self):
        """Mean function of the GP."""
        return self.measure.means[self]

    @property
    def name(self):
        """Name of the GP."""
        return self.measure[self]

    @name.setter
    @_dispatch
    def name(self, name: str):
        for measure in self._measures:
            measure.name(self, name)

    def __call__(self, x):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
            x (input): Points to construct the distribution at.

        Returns:
            :class:`.measure.FDD`: Finite-dimensional distribution.
        """
        return FDD(self, x)

    @_dispatch
    def __add__(self, other: Union[B.Numeric, FunctionType]):
        res = GP()
        for measure in self._measures:
            measure.sum(res, self, other)
        return res

    @_dispatch
    def __add__(self, other: "GP"):
        res = GP()
        for measure in intersection_measure_group(self, other):
            measure.sum(res, self, other)
        return res

    @_dispatch
    def __mul__(self, other: Union[B.Numeric, FunctionType]):
        res = GP()
        for measure in self._measures:
            measure.mul(res, self, other)
        return res

    @_dispatch
    def __mul__(self, other: "GP"):
        res = GP()
        for measure in intersection_measure_group(self, other):
            measure.mul(res, self, other)
        return res

    def shift(self, shift):
        """Shift the GP. See :meth:`.measure.Graph.shift`."""
        res = GP()
        for measure in self._measures:
            measure.shift(res, self, shift)
        return res

    def stretch(self, stretch):
        """Stretch the GP. See :meth:`.measure.Graph.stretch`."""
        res = GP()
        for measure in self._measures:
            measure.stretch(res, self, stretch)
        return res

    def __gt__(self, stretch):
        """Shorthand for :meth:`.measure.GP.stretch`."""
        return self.stretch(stretch)

    def transform(self, f):
        """Input transform the GP. See :meth:`.measure.Graph.transform`."""
        res = GP()
        for measure in self._measures:
            measure.transform(res, self, f)
        return res

    def select(self, *dims):
        """Select dimensions from the input. See :meth:`.measure.Graph.select`."""
        res = GP()
        for measure in self._measures:
            measure.select(res, self, *dims)
        return res

    def __getitem__(self, *dims):
        """Shorthand for :meth:`.measure.GP.select`."""
        return self.select(*dims)

    def diff(self, dim=0):
        """Differentiate the GP. See :meth:`.measure.Graph.diff`."""
        res = GP()
        for measure in self._measures:
            measure.diff(res, self, dim)
        return res

    def diff_approx(self, deriv=1, order=6):
        """Approximate the derivative of the GP by constructing a finite
        difference approximation.

        Args:
            deriv (int, optional): Order of the derivative. Defaults to `1`.
            order (int, optional): Order of the estimate. Defaults to `6`.

        Returns:
            :class:`.measure.GP`: Approximation of the derivative of the GP.
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
        if self._measures:
            return (
                f"GP({self.mean.display(formatter)}, {self.kernel.display(formatter)})"
            )
        else:
            return "GP()"


PromisedGP.deliver(GP)


def cross(*ps):
    """Construct the Cartesian product of a collection of processes.

    Args:
        *ps (:class:`.measure.GP`): Processes to construct the Cartesian product of.

    Returns:
        :class:`.measure.GP`: The Cartesian product of `ps`.
    """
    p_cross = GP()
    for measure in intersection_measure_group(*ps):
        measure.cross(p_cross, *ps)
    return p_cross


class FDD(Normal):
    """Finite-dimensional distribution.

    Args:
        p (:class:`.random.RandomProcess`): Process of FDD.
        x (input): Inputs that `p` is evaluated at.

    Attributes:
        p (:class:`.random.RandomProcess`): Process of FDD.
        x (input): Inputs that `p` is evaluated at.
    """

    def __init__(self, p, x):
        self.p = p
        self.x = x
        Normal.__init__(self, lambda: p.mean(x), lambda: p.kernel(x))

    def __str__(self):
        return f"FDD({self.p}, {self.x})"

    def __repr__(self):
        return f"FDD({self.p!r}, {self.x!r})"


PromisedFDD.deliver(FDD)


@num_elements.dispatch
def num_elements(fdd: FDD):
    return num_elements(fdd.x)
