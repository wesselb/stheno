# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, RQ, NoisyKernel, Observed, Kronecker
from stheno.input import Latent

# Define the grid for which we are going to generate function values.
x_true = np.linspace(0, 1, 500)[:, None]

# Define a GP that will generate the function values.
p_true = GP(kernel=RQ(1e-1).stretch(2).periodic(.3))

# Generate the function values for the grid.
y_true = p_true(x_true).sample()

# Now throw away approximately 80% of the generated function values.
n = np.shape(x_true)[0]
inds = np.random.choice(n, int(np.round(.2 * n)), replace=False)
x, y = x_true[inds, :], y_true[inds, :]

# Add some noise.
noise = .1
y += noise ** .5 * np.random.randn(*np.shape(y))

# Perform inference.
p = GP(NoisyKernel(p_true.kernel, noise * Kronecker()))
p_post = p.condition(Observed(x), y)

# Perform prediction.
pred_mean, pred_std = p_post.predict(Latent(x_true))

# Plot the results.
x_true, y_true = x_true.squeeze(), y_true.squeeze()
pred_mean, pred_std = pred_mean.squeeze(), pred_std.squeeze()

plt.plot(x_true, y_true, label='True', c='tab:blue')
plt.scatter(x.squeeze(), y.squeeze(), label='Observations', c='tab:red')
plt.plot(x_true, pred_mean, label='Prediction', c='tab:green')
plt.plot(x_true, pred_mean + 2 * pred_std, ls='--', c='tab:green')
plt.plot(x_true, pred_mean - 2 * pred_std, ls='--', c='tab:green')
plt.legend()
plt.show()
