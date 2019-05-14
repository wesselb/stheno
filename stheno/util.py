# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from functools import wraps
import logging
from numbers import Number
from time import time
from types import FunctionType

from lab import B
from plum import Dispatcher, Self, Referentiable

__all__ = []

_dispatch = Dispatcher()


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

    # Now check the rank of `x` and act accordingly.
    rank = B.rank(x)
    if rank > 2:
        raise ValueError('Input must be at most rank 2.')
    elif rank == 2:
        return x
    elif rank == 1:
        return B.expand_dims(x, axis=1)
    else:
        # Rank must be 0.
        return B.expand_dims(B.expand_dims(x, axis=0), axis=1)


@_dispatch(FunctionType)
def uprank(f):
    """A decorator to ensure that the rank of the arguments is two."""

    @wraps(f)
    def wrapped_f(*args):
        return f(*[uprank(x) for x in args])

    return wrapped_f
