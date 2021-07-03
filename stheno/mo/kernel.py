from lab import B
from mlkernels import Kernel, pairwise, elwise, num_elements

from .. import PromisedFDD as FDD, _dispatch
from .input import infer_size

__all__ = ["MultiOutputKernel"]


class MultiOutputKernel(Kernel):
    """A generic multi-output kernel.

    Args:
        measure (:class:`stheno.model.measure.Measure`): Measure to take the kernels
            from.
        *ps (:class:`stheno.model.gp.GP`): Processes that make up the multi-valued
            process.

    Attributes:
        measure (:class:`stheno.model.measure.Measure`): Measure to take the kernels
            from.
        ps (tuple[:class:`stheno.model.gp.GP`]): Processes that make up the
            multi-valued process.
    """

    def __init__(self, measure, *ps):
        self.measure = measure
        self.ps = ps

    def render(self, formatter):
        ks = [str(self.measure.kernels[p]) for p in self.ps]
        return "MultiOutputKernel({})".format(", ".join(ks))


@pairwise.dispatch
def pairwise(k: MultiOutputKernel, x: B.Numeric, y: B.Numeric):
    return pairwise(k, tuple(p(x) for p in k.ps), tuple(p(y) for p in k.ps))


@pairwise.dispatch
def pairwise(k: MultiOutputKernel, x: FDD, y: B.Numeric):
    return pairwise(k, (x,), tuple(p(y) for p in k.ps))


@pairwise.dispatch
def pairwise(k: MultiOutputKernel, x: B.Numeric, y: FDD):
    return pairwise(k, tuple(p(x) for p in k.ps), (y,))


@pairwise.dispatch
def pairwise(k: MultiOutputKernel, x: FDD, y: FDD):
    return k.measure.kernels[x.p, y.p](x.x, y.x)


@elwise.dispatch
def elwise(k: MultiOutputKernel, x: B.Numeric, y: B.Numeric):
    return elwise(k, tuple(p(x) for p in k.ps), tuple(p(y) for p in k.ps))


@elwise.dispatch
def elwise(k: MultiOutputKernel, x: FDD, y: B.Numeric):
    raise ValueError('Unclear combination of arguments given to "elwise".')


@elwise.dispatch
def elwise(k: MultiOutputKernel, x: B.Numeric, y: FDD):
    raise ValueError('Unclear combination of arguments given to "elwise".')


@elwise.dispatch
def elwise(k: MultiOutputKernel, x: FDD, y: FDD):
    return k.measure.kernels[x.p, y.p].elwise(x.x, y.x)


@_dispatch
def dimensionality(k: MultiOutputKernel):
    return len(k.ps)


@_dispatch
def _take_x(k: MultiOutputKernel, x: B.Numeric, mask: B.Numeric):
    i = 0
    x_taken = ()
    for p in k.ps:
        n = infer_size(k, p(x))
        x_taken += (_take_x(k, p(x), mask[i : i + n]),)
        i += n
    return x_taken


@_dispatch
def _take_x(k: MultiOutputKernel, x: FDD, mask: B.Numeric):
    if x.p not in k.ps:
        raise ValueError(f"Process {x.p} is not part of the multi-output kernel.")
    return B.take(x, mask)
