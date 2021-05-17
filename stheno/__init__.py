# noinspection PyUnresolvedReferences
import matrix
# noinspection PyUnresolvedReferences
from mlkernels import *
# noinspection PyUnresolvedReferences
import lab as B

from plum import PromisedType, Dispatcher

PromisedFDD = PromisedType()
PromisedGP = PromisedType()
PromisedMeasure = PromisedType()

_dispatch = Dispatcher()


class BreakingChangeWarning(UserWarning):
    """A breaking change."""


from .lazy import *
from .model import *
from .mo import *
from .random import *
