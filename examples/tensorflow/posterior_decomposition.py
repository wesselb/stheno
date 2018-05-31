# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lab.tf import B
from stheno.tf import GP, EQ, RQ, Observed, AdditiveComponentKernel, \
    Component, Kronecker, Latent

B.epsilon = 1e-8

# Start a TensorFlow session.
s = tf.Session()

# Define the grid for which we are going to generate function values.
x = np.linspace(0, 2, 500)[:, None]

# Define a GP that will generate the function values.
k = AdditiveComponentKernel({
    Component('smooth'): RQ(1e-1).stretch(0.3),
    Component('periodic'): EQ().periodic(.3),
    Component('wiggly'): EQ().stretch(0.02),
    Component('noise'): 0.1 * Kronecker()
    # Specify that the latent function consists of the sum of the below
    # components.
}, latent=[Component('smooth'), Component('periodic'), Component('wiggly')])
p = GP(k)

# Generate the function values for the grid.
y_smooth = s.run(p(Component('smooth')(x)).sample())
y_periodic = s.run(p(Component('periodic')(x)).sample())
y_wiggly = s.run(p(Component('wiggly')(x)).sample())
y_noise = s.run(p(Component('noise')(x)).sample())
y = y_smooth + y_periodic + y_wiggly + y_noise

# Now throw away approximately 80% of the generated function values.
n = np.shape(x)[0]
inds = np.random.choice(n, int(np.round(.2 * n)), replace=False)
x_obs, y_obs = x[inds, :], y[inds, :]

# Show predictions.
p_post = p.condition(Observed(x_obs), y_obs)


def plot(x_true, y_true, x_obs, y_obs, pred_mean, pred_std):
    x_true, y_true = x_true.squeeze(), y_true.squeeze()
    pred_mean, pred_std = pred_mean.squeeze(), pred_std.squeeze()
    plt.plot(x_true, y_true, label='True', c='tab:blue')
    if x_obs is not None and y_obs is not None:
        plt.scatter(x_obs.squeeze(), y_obs.squeeze(),
                    label='Observations', c='tab:red')
    plt.plot(x_true, pred_mean, label='Prediction', c='tab:green')
    plt.plot(x_true, pred_mean + 2 * pred_std, ls='--', c='tab:green')
    plt.plot(x_true, pred_mean - 2 * pred_std, ls='--', c='tab:green')
    plt.legend()


# Plot the results.
plt.subplot(2, 2, 1)
plot(x, y, x_obs, y_obs, *s.run(p_post.predict(Latent(x))))
plt.subplot(2, 2, 2)
plot(x, y_smooth, None, None, *s.run(p_post.predict(Component('smooth')(x))))
plt.subplot(2, 2, 3)
plot(x, y_periodic, None, None,
     *s.run(p_post.predict(Component('periodic')(x))))
plt.subplot(2, 2, 4)
plot(x, y_wiggly, None, None, *s.run(p_post.predict(Component('wiggly')(x))))
plt.show()
