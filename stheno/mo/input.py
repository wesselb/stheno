import lab as B
from mlkernels import pairwise, elwise

__all__ = []


@pairwise.dispatch(precedence=1)
def pairwise(k, x: tuple, y: tuple):
    return B.block(*[[pairwise(k, xi, yi) for yi in y] for xi in x])


@pairwise.dispatch(precedence=1)
def pairwise(k, x: tuple, y):
    return pairwise(k, x, (y,))


@pairwise.dispatch(precedence=1)
def pairwise(k, x, y: tuple):
    return pairwise(k, (x,), y)


@elwise.dispatch(precedence=1)
def elwise(k, x: tuple, y: tuple):
    if len(x) != len(y):
        raise ValueError('"elwise" must be called with similarly sized tuples.')
    return B.concat(*[elwise(k, xi, yi) for xi, yi in zip(x, y)], axis=-2)


@elwise.dispatch(precedence=1)
def elwise(k, x: tuple, y):
    return elwise(k, x, (y,))


@elwise.dispatch(precedence=1)
def elwise(k, x, y: tuple):
    return elwise(k, (x,), y)
