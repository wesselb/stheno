import logging

from lab import B
from plum import Dispatcher

from . import PromisedFDD as FDD
from .input import MultiInput
from .mean import Mean

__all__ = ["MultiOutputMean"]

_dispatch = Dispatcher()

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

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    @_dispatch
    def __call__(self, x: B.Numeric):
        return self(MultiInput(*(p(x) for p in self.ps)))

    @_dispatch
    def __call__(self, x: FDD):
        return self.measure.means[x.p](x.x)

    @_dispatch
    def __call__(self, x: MultiInput):
        return B.concat(*[self(xi) for xi in x.get()], axis=0)

    def render(self, formatter):
        ms = [str(self.measure.means[p]) for p in self.ps]
        return "MultiOutputMean({})".format(", ".join(ms))
