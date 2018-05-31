# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

__all__ = ['Input', 'Observed', 'Latent']


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


class Observed(Input):
    """Observed points."""


class Latent(Input):
    """Latent points."""
