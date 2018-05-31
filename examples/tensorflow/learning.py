# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI

from stheno.tf import GP, EQ, Kronecker, Observed, Latent, \
    AdditiveComponentKernel, Component


# Define the model kernel structure.
def NoisyKernel(k1, k2):
    return AdditiveComponentKernel({Latent: k1, Component('noise'): k2})


# Start a TensorFlow session.
s = tf.Session()

# Define the grid for which we are going to generate function values.
x_true = np.linspace(0, 1, 500)[:, None]

# Define a GP that will generate the function values.
true_scale = .05
true_noise = 0.05
p_true = GP(NoisyKernel(EQ().stretch(true_scale), true_noise * Kronecker()))

# Generate the function values for the grid.
y_true = s.run(p_true(Latent(x_true)).sample())

# Generate observations.
y_obs = y_true + s.run(p_true(Component('noise')(x_true)).sample())

# Now throw away approximately 80% of the generated function values.
n = np.shape(x_true)[0]
inds = np.random.choice(n, int(np.round(.2 * n)), replace=False)
x, y = x_true[inds, :], y_obs[inds, :]

# Create a positive TensorFlow variable for the length scale of a GP that we
# are going to fit to the noisy data.
log_scale = tf.Variable(np.log(0.1))
scale = tf.exp(log_scale)

# Create another positive TensorFlow variable for the variance of the noise of
# the data.
log_noise = tf.Variable(np.log(0.01))
noise = tf.exp(log_noise)

# Initialise the variables.
s.run(tf.variables_initializer([log_scale, log_noise]))

# Create the GP that we are going to fit to the data.
p = GP(NoisyKernel(EQ().stretch(scale), noise * Kronecker()))

# Optimise the marginal likelihood with respect to the length scale of the
# kernel.
lml = p(Observed(x)).log_pdf(y)
SOI(-lml, options={'disp': True}).minimize(s)

# Print the true and inferred length scale.
print('Length scale:')
print('  True: {:.3f}'.format(true_scale))
print('  Inferred: {:.3f}'.format(s.run(scale)))

# Finally print the true and inferred noise variance.
print('Noise variance:')
print('  True: {:.3f}'.format(true_noise))
print('  Inferred: {:.3f}'.format(s.run(noise)))

# Show predictions.
p_post = p.condition(Observed(x), y)

# Perform prediction.
pred_mean, pred_std = s.run(p_post.predict(Latent(x_true)))

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
