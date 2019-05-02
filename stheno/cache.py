# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from numbers import Number
from time import time
from types import FunctionType

from lab import B
from plum import Dispatcher, Self, Referentiable

__all__ = ['Cache', 'cache']

log_cache_call = logging.getLogger(__name__ + '.call')
log_cache_lab = logging.getLogger(__name__ + '.lab')

_dispatch = Dispatcher()


class Cache(Referentiable):
    """Cache for kernels and means.

    Caches output of calls. Also caches calls to `B.*`: call instead `cache.*`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self):
        self._cache_call = {}
        self._cache_lab = {}
        self._start = time()
        self._dump = []
        self.depth = 0

    def dump(self, *objs):
        """A dump for objects to prevent the GC from cleaning them up.

        Args:
            *objs (object): Objects to dump.
        """
        self._dump.append(objs)

    @_dispatch(object)
    def _resolve(self, key):
        return id(key)

    @_dispatch({Number, str, bool})
    def _resolve(self, key):
        return key

    @_dispatch({tuple, list})
    def _resolve(self, key):
        return tuple(self._resolve(x) for x in key)

    def __getitem__(self, key):
        return self._cache_call[self._resolve(key)]

    def __setitem__(self, key, output):
        # Prevent GC from corrupting cache!
        self.dump(key)
        self._cache_call[self._resolve(key)] = output

    def __getattr__(self, f):
        attr = getattr(B, f)

        # Simply return if `attr` is not callable.
        if not callable(attr):
            return attr

        # Otherwise, return a wrapper.
        def call_cached(*args, **kw_args):
            # Let the key depend on the function name...
            key = (f,)

            # ...on the arguments...
            key += self._resolve(args)

            # ...and on the keyword arguments.
            if len(kw_args) > 0:
                # Sort keyword arguments according to keys.
                items = tuple(sorted(kw_args.items(), key=lambda x: x[0]))
                key += self._resolve(items)

            # Attempt cached execution.
            try:
                out = self._cache_lab[key]
                log_cache_lab.debug('%4.0f ms: Hit for "%s" with key "%s".',
                                    self.life_ms(), f, key)
                return out
            except KeyError:
                pass

            # Filter keyword `cache_id`.
            try:
                del kw_args['cache_id']
            except KeyError:
                pass

            # Prevent GC from corrupting cache!
            self.dump(f, args, kw_args)

            self._cache_lab[key] = attr(*args, **kw_args)
            return self._cache_lab[key]

        return call_cached

    def life_ms(self):
        """Get the number of milliseconds since the creation of the cache."""
        return (time() - self._start) * 1e3


def cache(f):
    """A decorator for methods to cache their outputs."""

    name = f.__name__

    def wrapped_f(self, *args):
        inputs, cache = args[:-1], args[-1]
        try:
            out = cache[self, f, inputs]
            log_cache_call.debug('%4.0f ms: Hit for "%s".',
                                 cache.life_ms(), type(self).__name__)
            return out
        except KeyError:
            pass

        # Log and increase depth.
        log_cache_call.debug('%4.0f ms: Miss for "%s": start; depth: %d.',
                             cache.life_ms(), type(self).__name__, cache.depth)
        cache.depth += 1

        # Perform execution.
        cache[self, f, inputs] = f(self, *args)

        # Log and decrease depth.
        log_cache_call.debug('%4.0f ms: Miss for "%s": end.',
                             cache.life_ms(), type(self).__name__)
        cache.depth -= 1

        return cache[self, f, inputs]

    wrapped_f.__name__ = name

    return wrapped_f


@_dispatch(object, [object])
def uprank(x, B=B):
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
    if B.rank(x) > 2:
        raise ValueError('Input must be at most rank 2.')
    elif B.rank(x) == 2:
        return x
    elif B.rank(x) == 1:
        return B.expand_dims(x, axis=1)
    else:
        # Rank must be 0.
        return B.expand_dims(B.expand_dims(x, axis=0), axis=1)


@_dispatch(FunctionType)
def uprank(f):
    """A decorator to ensure that the rank of the arguments is two."""

    def wrapped_f(self, *args):
        inputs, B = args[:-1], args[-1]
        return f(self, *([uprank(x, B) for x in inputs] + [B]))

    wrapped_f.__name__ = f.__name__

    return wrapped_f
