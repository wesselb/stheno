from functools import wraps
from types import FunctionType

from lab import B
from plum import Dispatcher

__all__ = ["num_elements", "uprank"]

_dispatch = Dispatcher()


@_dispatch(object)
def num_elements(x):
    """Determine the number of elements in an input.

    Deals with scalars, vectors, and matrices.

    Args:
        x (tensor): Input.

    Returns:
        int: Number of elements.
    """
    shape = B.shape(x)
    if shape == ():
        return 1
    else:
        return shape[0]


@_dispatch(object)
def uprank(x):
    """Ensure that the rank of `x` is 2.

    Args:
        x (tensor): Tensor to uprank.

    Returns:
        tensor: `x` with rank at least 2.
    """
    # Simply return non-numerical inputs.
    if not isinstance(x, B.Numeric):
        return x
    else:
        return B.uprank(x)


@_dispatch(FunctionType)
def uprank(f):
    """A decorator to ensure that the rank of the arguments is two."""

    @wraps(f)
    def wrapped_f(*args):
        return f(*[uprank(x) for x in args])

    return wrapped_f
