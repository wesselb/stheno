import logging

from lab import B
from plum import Dispatcher, Self

from . import PromisedFDD as FDD
from .input import MultiInput
from .mean import Mean

__all__ = ["MultiOutputMean"]

log = logging.getLogger(__name__)


class MultiOutputMean(Mean):
    """A generic multi-output mean.

    Args:
        measure (:class:`.measure.Measure`): Measure to take the means from.
        *ps (:class:`.graph.GP`): Processes that make up the multi-valued process.

    Attributes:
        measure (:class:`.measure.Measure`): Measure to take the means from.
        ps (tuple[:class:`.graph.GP`]): Processes that make up the multi-valued process.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    @_dispatch(B.Numeric)
    def __call__(self, x):
        return self(MultiInput(*(p(x) for p in self.ps)))

    @_dispatch(FDD)
    def __call__(self, x):
        return self.measure.means[x.p](x.x)

    @_dispatch(MultiInput)
    def __call__(self, x):
        return B.concat(*[self(xi) for xi in x.get()], axis=0)

    def render(self, formatter):
        ms = [str(self.measure.means[p]) for p in self.ps]
        return "MultiOutputMean({})".format(", ".join(ms))
