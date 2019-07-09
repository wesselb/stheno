# -*- coding: utf-8 -*-

from __future__ import absolute_import  # pragma: no cover

import logging

# noinspection PyUnresolvedReferences
import lab.tensorflow
from plum import Dispatcher
from tensorflow import Tensor

# noinspection PyUnresolvedReferences
from . import *  # pragma: no cover
from .field import Element
from .random import Random

log = logging.getLogger(__name__)

_dispatch = Dispatcher()

# Save original methods.
__mul__tensor = Tensor.__mul__
__add__tensor = Tensor.__add__


@_dispatch(Tensor, object)
def __mul__(self, other):
    return __mul__tensor(self, other)


@_dispatch(Tensor, {Element, Random})
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(Tensor, object)
def __add__(self, other):
    return __add__tensor(self, other)


@_dispatch(Tensor, {Element, Random})
def __add__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
Tensor.__mul__ = __mul__
Tensor.__add__ = __add__
