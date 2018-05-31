# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import parametric

__all__ = ['Input', 'Observed', 'Latent', 'Component']


class Input(object):
    """Input type.

    Args:
        x (tensor): Input to type.
    """

    def __init__(self, x):
        self._x = x

    def get(self):
        """Get the wrapped input."""
        return self._x


@parametric
class Component(Input):
    """A particular component.

    This is a parametric type."""


Observed = Component('observed')  #: Observed points
Latent = Component('latent')  #: Latent points
