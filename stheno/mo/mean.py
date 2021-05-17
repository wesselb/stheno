from lab import B
from mlkernels import Mean

from .. import PromisedFDD as FDD, _dispatch

__all__ = ["MultiOutputMean"]


class MultiOutputMean(Mean):
    """A generic multi-output mean.

    Args:
        measure (:class:`stheno.model.measure.Measure`): Measure to take the means from.
        *ps (:class:`stheno.model.gp.GP`): Processes that make up the multi-valued
            process.

    Attributes:
        measure (:class:`stheno.model.measure.Measure`): Measure to take the means from.
        ps (tuple[:class:`stheno.model.gp.GP`]): Processes that make up the
            multi-valued process.
    """

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    @_dispatch
    def __call__(self, x: B.Numeric):
        return self(tuple(p(x) for p in self.ps))

    @_dispatch
    def __call__(self, x: FDD):
        return self.measure.means[x.p](x.x)

    @_dispatch
    def __call__(self, x: tuple):
        return B.concat(*[self(xi) for xi in x], axis=0)

    def render(self, formatter):
        ms = [str(self.measure.means[p]) for p in self.ps]
        return "MultiOutputMean({})".format(", ".join(ms))


@_dispatch
def dimensionality(m: MultiOutputMean):
    return len(m.ps)
