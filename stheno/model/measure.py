from types import FunctionType

from lab import B
from matrix import Constant
from mlkernels import (
    num_elements,
    ZeroKernel,
    TensorProductKernel,
)
from plum import Union

from .fdd import FDD
from .gp import GP, assert_same_measure
from .observations import (
    AbstractObservations,
    Observations,
    PseudoObservations,
    combine,
)
from .. import _dispatch, PromisedMeasure
from ..lazy import LazyVector, LazyMatrix
from ..mo import MultiOutputKernel as MOK, MultiOutputMean as MOM
from ..random import Normal

__all__ = ["Measure"]


class Measure:
    """A GP model.

    Attributes:
        ps (list[:class:`.gp.GP`]): Processes of the measure.
        mean (:class:`stheno.lazy.LazyVector`): Mean.
        kernels (:class:`stheno.lazy.LazyMatrix`): Kernels.
        default (:class:`.measure.Measure` or None): Global default measure.
    """
    default = None

    def __init__(self):
        self.ps = []
        self._pids = set()
        self.means = LazyVector()
        self.kernels = LazyMatrix()

        # Store named GPs in both ways.
        self._gps_by_name = {}
        self._names_by_gp = {}

        self._prev_default = None

    def __enter__(self):
        self._prev_default = self.default
        Measure.default = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Measure.default = self._prev_default

    def __hash__(self):
        # This is needed for :func:`.gp.intersection_measure_group`, which puts
        # many :class:`.measure.Measure`s in a `set`. Every measure is unique.
        return id(self)

    @_dispatch
    def __getitem__(self, name: str):
        return self._gps_by_name[name]

    @_dispatch
    def __getitem__(self, p: GP):
        return self._names_by_gp[id(p)]

    @_dispatch
    def name(self, p: GP, name: str):
        """Name a GP.

        Args:
            p (:class:`.gp.GP`): GP to name.
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
    def __call__(self, p: GP):
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
    def __call__(self, fdd: FDD):
        return self(fdd.p)(fdd.x, fdd.noise)

    def add_independent_gp(self, p, mean, kernel):
        """Add an independent GP to the model.

        Args:
            p (:class:`.gp.GP`): GP to add.
            mean (:class:`mlkernels.Mean`): Mean function of GP.
            kernel (:class:`mlkernels.Kernel`): Kernel function of GP.

        Returns:
            :class:`.gp.GP`: The newly added independent GP.
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
    def sum(self, p_sum: GP, other, p: GP):
        """Sum a GP from the graph with another object.

        Args:
            p_sum (:class:`.gp.GP`): GP that is the sum.
            obj1 (other type or :class:`.gp.GP`): First term in the sum.
            obj2 (other type or :class:`.gp.GP`): Second term in the sum.

        Returns:
            :class:`.gp.GP`: The GP corresponding to the sum.
        """
        return self.sum(p_sum, p, other)

    @_dispatch
    def sum(self, p_sum: GP, p: GP, other: Union[B.Numeric, FunctionType]):
        return self._update(
            p_sum,
            self.means[p] + other,
            self.kernels[p],
            lambda j: self.kernels[p, j],
        )

    @_dispatch
    def sum(self, p_sum: GP, p1: GP, p2: GP):
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
    def mul(self, p_mul: GP, other, p: GP):
        """Multiply a GP from the graph with another object.

        Args:
            p_mul (:class:`.gp.GP`): GP that is the product.
            obj1 (object): First factor in the product.
            obj2 (object): Second factor in the product.
            other (object): Other object in the product.

        Returns:
            :class:`.gp.GP`: The GP corresponding to the product.
        """
        return self.mul(p_mul, p, other)

    @_dispatch
    def mul(self, p_mul: GP, p: GP, other: B.Numeric):
        return self._update(
            p_mul,
            self.means[p] * other,
            self.kernels[p] * other ** 2,
            lambda j: self.kernels[p, j] * other,
        )

    @_dispatch
    def mul(self, p_mul: GP, p: GP, f: FunctionType):
        def ones(x):
            return Constant(B.one(x), num_elements(x), 1)

        return self._update(
            p_mul,
            f * self.means[p],
            f * self.kernels[p],
            lambda j: TensorProductKernel(f, ones) * self.kernels[p, j],
        )

    @_dispatch
    def mul(self, p_mul: GP, p1: GP, p2: GP):
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
            p_shifted (:class:`.gp.GP`): Shifted GP.
            p (:class:`.gp.GP`): GP to shift.
            shift (object): Amount to shift by.

        Returns:
            :class:`.gp.GP`: The shifted GP.
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
            p_stretched (:class:`.gp.GP`): Stretched GP.
            p (:class:`.gp.GP`): GP to stretch.
            stretch (object): Extent of stretch.

        Returns:
            :class:`.gp.GP`: The stretched GP.
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
            p_selected (:class:`.gp.GP`): GP with selected inputs.
            p (:class:`.gp.GP`): GP to select input dimensions from.
            *dims (object): Dimensions to select.

        Returns:
            :class:`.gp.GP`: GP with the specific input dimensions.
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
            p_transformed (:class:`.gp.GP`): GP with transformed inputs.
            p (:class:`.gp.GP`): GP to input transform.
            f (function): Input transform.

        Returns:
            :class:`.gp.GP`: Input-transformed GP.
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
            p_diff (:class:`.gp.GP`): Derivative.
            p (:class:`.gp.GP`): GP to differentiate.
            dim (int, optional): Dimension of feature which to take the derivative
                with respect to. Defaults to `0`.

        Returns:
            :class:`.gp.GP`: Derivative of GP.
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
            obs (:class:`.observations.AbstractObservations`): Observations to condition on.

        Returns:
            list[:class:`.gp.GP`]: Posterior processes.
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
    def condition(self, fdd: FDD, y: B.Numeric):
        return self.condition(Observations(fdd, y))

    @_dispatch
    def condition(self, pair: tuple):
        return self.condition(Observations(*pair))

    @_dispatch
    def condition(self, *pairs: tuple):
        return self.condition(Observations(*pairs))

    @_dispatch
    def __or__(self, *args):
        return self.condition(*args)

    @_dispatch
    def cross(self, p_cross: GP, *ps: GP):
        """Construct the Cartesian product of a collection of processes.

        Args:
            p_cross (:class:`.gp.GP`): GP that is the Cartesian product.
            *ps (:class:`.gp.GP`): Processes to construct the Cartesian product of.

        Returns:
            :class:`.gp.GP`: The Cartesian product of `ps`.
        """
        mok = MOK(self, *ps)
        return self._update(
            p_cross,
            MOM(self, *ps),
            mok,
            lambda j: mok.transform(None, lambda y: FDD(j, y)),
        )

    @_dispatch
    def sample(self, n: int, *fdds: FDD):
        """Sample multiple processes simultaneously.

        Args:
            n (int, optional): Number of samples. Defaults to `1`.
            *fdds (:class:`.fdd.FDD`): Locations to sample at.

        Returns:
            tuple: Tuple of samples.
        """
        sample = combine(*fdds).sample(n)

        # Unpack sample.
        lengths = [num_elements(fdd) for fdd in fdds]
        i, samples = 0, []
        for length in lengths:
            samples.append(sample[i : i + length, :])
            i += length
        return samples[0] if len(samples) == 1 else samples

    @_dispatch
    def sample(self, *fdds: FDD):
        return self.sample(1, *fdds)

    @_dispatch
    def logpdf(self, *pairs: Union[list, tuple]):
        """Compute the logpdf of one multiple observations.

        Can also give an `AbstractObservations`.

        Args:
            *pairs (tuple[:class:`.fdd.FDD`, tensor]): Pairs of FDDs and values
                of the observations.

        Returns:
            scalar: Logpdf.
        """
        fdd, y = combine(*pairs)
        return self(fdd).logpdf(y)

    @_dispatch
    def logpdf(self, fdd: FDD, y: B.Numeric):
        return self(fdd).logpdf(y)

    @_dispatch
    def logpdf(self, obs: Observations):
        return self.logpdf(obs.fdd, obs.y)

    @_dispatch
    def logpdf(self, obs: PseudoObservations):
        return obs.elbo(self)


PromisedMeasure.deliver(Measure)
