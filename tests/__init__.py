import os
import sys

# Add package to path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..')))

# Load LAB extensions.
# noinspection PyUnresolvedReferences
import lab.autograd
# noinspection PyUnresolvedReferences
import lab.torch
# noinspection PyUnresolvedReferences
import lab.tensorflow
# noinspection PyUnresolvedReferences
import lab.jax

# Load Stheno extensions.
# noinspection PyUnresolvedReferences
import stheno.autograd
# noinspection PyUnresolvedReferences
import stheno.torch
# noinspection PyUnresolvedReferences
import stheno.tensorflow
# noinspection PyUnresolvedReferences
import stheno.jax
