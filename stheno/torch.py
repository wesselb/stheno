# -*- coding: utf-8 -*-

from __future__ import absolute_import  # pragma: no cover

import logging

from plum import Dispatcher
from torch import Tensor

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .field import Element

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

B.backend_to_torch()  # pragma: no cover

# Save original methods.
__mul__tensor = Tensor.__mul__
__add__tensor = Tensor.__add__


@_dispatch(Tensor, object)
def __mul__(self, other):
    return __mul__tensor(self, other)


@_dispatch(Tensor, Element)
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(Tensor, object)
def __add__(self, other):
    return __add__tensor(self, other)


@_dispatch(Tensor, Element)
def __add__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
Tensor.__mul__ = __mul__
Tensor.__add__ = __add__
