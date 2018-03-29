# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI

from stheno import GP, EQ, Noise
from lab import B

B.backend_to_tf()

# Start a TensorFlow session.
s = tf.Session()

# Define the grid for which we are going to generate function values.
x = np.array([np.linspace(0, 1, 100)])

# Define a GP that will generate the function values.
true_scale = .05
gp = GP(kernel=EQ().stretch(true_scale))

# Generate the function values for the grid.
y = s.run(gp(x).sample())

# Add some noise.
true_noise = 0.05
y += true_noise ** .5 * np.random.randn(*np.shape(y))

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
kernel = EQ().stretch(scale) + noise * Noise()
gp = GP(kernel=kernel)

# Optimise the marginal likelihood with respect to the length scale of the
# kernel.
lml = gp(x).log_pdf(y)
SOI(-lml).minimize(s)

# Print the true and inferred length scale.
print('Length scale:')
print('  True: {:.3f}'.format(true_scale))
print('  Inferred: {:.3f}'.format(s.run(scale)))

# Finally print the true and inferred noise variance.
print('Noise variance:')
print('  True: {:.3f}'.format(true_noise))
print('  Inferred: {:.3f}'.format(s.run(noise)))
