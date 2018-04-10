# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, RQ

# Define the grid for which we are going to generate function values.
x_truth = np.linspace(0, 1, 500)[:, None]

# Define a GP that will generate the function values.
gp = GP(kernel=RQ(1e-1).stretch(2).periodic(.3))

# Generate the function values for the grid.
y_truth = gp(x_truth).sample()

# Now throw away approximately 80% of the generated function values.
n = np.shape(x_truth)[0]
inds = np.random.choice(n, int(np.round(.2 * n)), replace=False)
x, y = x_truth[inds, :], y_truth[inds, :]

# Add some noise.
noise = .1
y += noise ** .5 * np.random.randn(*np.shape(y))

# Perform inference.
gp_post = GP(kernel=gp.kernel).condition(x, y, noise=noise)

# Perform prediction.
pred_mean, pred_std = gp_post.predict(x_truth)

# Plot the results.
x_truth, y_truth = x_truth.squeeze(), y_truth.squeeze()
pred_mean, pred_std = pred_mean.squeeze(), pred_std.squeeze()

plt.plot(x_truth, y_truth, label='Truth', c='tab:blue')
plt.scatter(x.squeeze(), y.squeeze(), label='Observations', c='tab:red')
plt.plot(x_truth, pred_mean, label='Prediction', c='tab:green')
plt.plot(x_truth, pred_mean + 2 * pred_std, ls='--', c='tab:green')
plt.plot(x_truth, pred_mean - 2 * pred_std, ls='--', c='tab:green')
plt.legend()
plt.show()
