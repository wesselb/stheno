# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from stheno import B, GP, RQ, Matern12, Matern32, Matern52, EQ

# Add some regularisation.
B.epsilon = 1e-6

# Define the grid for which we are going to generate function values.
x = np.linspace(0, 5, 500)[:, None]

# Define a number of kernels.
kernels = [RQ(1e-1), Matern12(), Matern32(), Matern52(), EQ().periodic(1.)]
names = ['RQ', 'Matern--1/2', 'Matern--3/2', 'Matern--5/2', 'Periodic EQ']

# Generate a function for each of the defined kernels.
ys = []
for kernel in kernels:
    # Define a GP that will generate the function values.
    gp = GP(kernel)

    # Generate the function values for the grid.
    ys.append(gp(x).sample())

# Plot the results.
for name, y in zip(names, ys):
    plt.plot(x.squeeze(), y.squeeze(), label=name)
plt.legend()
plt.show()
