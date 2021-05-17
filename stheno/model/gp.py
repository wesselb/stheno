from types import FunctionType

from fdm import central_fdm
from lab import B
from mlkernels import (
    OneKernel,
    ZeroMean,
    OneMean,
)
from plum import Union

from .fdd import FDD
from .. import PromisedGP, PromisedMeasure, _dispatch
from ..random import RandomProcess

__all__ = ["assert_same_measure", "intersection_measure_group", "cross", "GP"]


def assert_same_measure(*ps):
    """Assert that processes are associated to the same measure.

    Args:
        *ps (:class:`.gp.GP`): Processes.
    """
    # First check that all processes are constructed from the same measure.
    for p in ps[1:]:
        if ps[0].measure != p.measure:
            raise AssertionError(
                f"Processes {ps[0]} and {p} are associated to different measures."
            )


def intersection_measure_group(*ps):
    """Get the intersection of the measures associated to a number of processes.

    Args:
        *ps (:class:`.gp.GP`): Processes.
    """
    assert_same_measure(*ps)
    intersection = set(ps[0]._measures)
    for p in ps[1:]:
        intersection &= set(p._measures)
    return intersection


def cross(*ps):
    """Construct the Cartesian product of a collection of processes.

    Args:
        *ps (:class:`.gp.GP`): Processes to construct the Cartesian product of.

    Returns:
        :class:`.gp.GP`: The Cartesian product of `ps`.
    """
    p_cross = GP()
    for measure in intersection_measure_group(*ps):
        measure.cross(p_cross, *ps)
    return p_cross


class GP(RandomProcess):
    """Gaussian process.

    Args:
        mean (:class:`mlkernels.Mean`, optional): Mean function of the process. Defaults
            to zero.
        kernel (:class:`mlkernels.Kernel`): Kernel of the process.
        measure (:class:`.measure.Measure`): Measure to attach to. Must be given as a
            keyword argument.
        name (str, optional): Name. Must be given as a keyword argument.
    """

    @_dispatch
    def __init__(self, mean, kernel, measure=None, name=None):
        self._measures = []

        if measure is None:
            # If no measure is given. See if a default it set.
            if PromisedMeasure.resolve().default:
                measure = PromisedMeasure.resolve().default
            else:
                measure = PromisedMeasure.resolve()()

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

    def __call__(self, x, noise=None):
        """Construct a finite-dimensional distribution at specified locations.

        Args:
            x (input): Points to construct the distribution at.
        noise (scalar, vector, or matrix, optional): Additive noise.

        Returns:
            :class:`.fdd.FDD`: Finite-dimensional distribution.
        """
        return FDD(self, x, noise)

    def condition(self, *args):
        """Condition `self.measure` on data and obtain the posterior GP.

        See :meth:`.measure.Measure.condition` for a description of the arguments.

        Returns:
            :class:`.gp.GP`: Posterior GP.
        """
        posterior = self.measure.condition(*args)
        return posterior(self)

    @_dispatch
    def __or__(self, *args):
        """Shorthand for :meth:`.gp.GP.condition`."""
        return self.condition(*args)

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
        """Shift the GP. See :meth:`.measure.Measure.shift`."""
        res = GP()
        for measure in self._measures:
            measure.shift(res, self, shift)
        return res

    def stretch(self, stretch):
        """Stretch the GP. See :meth:`.measure.Measure.stretch`."""
        res = GP()
        for measure in self._measures:
            measure.stretch(res, self, stretch)
        return res

    def transform(self, f):
        """Input transform the GP. See :meth:`.measure.Measure.transform`."""
        res = GP()
        for measure in self._measures:
            measure.transform(res, self, f)
        return res

    def select(self, *dims):
        """Select dimensions from the input. See :meth:`.measure.Measure.select`."""
        res = GP()
        for measure in self._measures:
            measure.select(res, self, *dims)
        return res

    def diff(self, dim=0):
        """Differentiate the GP. See :meth:`.measure.Measure.diff`."""
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
            :class:`.gp.GP`: Approximation of the derivative of the GP.
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
