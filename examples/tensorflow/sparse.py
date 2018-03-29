# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI

from stheno.tf import B, GP, EQ, SPD, PosteriorKernel, Normal, Noise

B.default_reg_diag = 1e-8

# Start a TensorFlow session.
s = tf.Session()

# Define the grid for which we are going to generate function values.
x = np.array([np.linspace(0, 1, 100)])
n_u = 5
x_u = np.array([np.linspace(0, 1, n_u)])

# Define a GP that will generate the function values.
true_scale = .05
gp = GP(kernel=EQ().stretch(true_scale))

# Generate the function values for the grid.
y = s.run(gp(x).sample())

n = np.shape(x)[1]
inds = np.random.choice(n, int(np.round(.5 * n)), replace=False)
x_obs, y_obs = x[:, inds], y[inds, :]

# Add some noise.
noise = .02
y_obs += noise ** .5 * np.random.randn(*np.shape(y_obs))

# Condition.
gp = gp.condition(x_obs, y_obs, noise)

# Construct approximate posterior.
log_scale = tf.Variable(np.log(0.2))
scale = tf.exp(log_scale)
log_var = tf.Variable(np.log(1))
var = tf.exp(log_var)
s.run(tf.variables_initializer([log_scale, log_var]))

# Create the GP that we are going to fit to the data.
q_gp = GP(kernel=var * EQ().stretch(scale) + 1e-6 * Noise())
K = SPD(q_gp.kernel(x_u))
q_L = tf.Variable(s.run(K.cholesky()))
q_mu2 = tf.Variable(np.zeros((n_u, 1)))
s.run(tf.variables_initializer([q_L, q_mu2]))

A = be.transpose(K.inv_prod(q_gp.kernel(x_u, x)))
Sig = PosteriorKernel(q_gp, x_u, K)(x)
q = A * Normal(be.dot(q_L, q_L, tr_a=True), q_mu2) + Normal(Sig)

p = gp(x)
p_mu, p_std = s.run(p.mean).squeeze(), np.diag(s.run(p.var.mat)) ** .5
kl = q.kl(p)
w2 = q.w2(p)

print('KL before minimisation: {}'.format(s.run(kl)))
print('W2 before minimisation: {}'.format(s.run(w2)))
print('Scale before minimisation: {}'.format(s.run(scale)))
print('Var before minimisation: {}\n'.format(s.run(var)))

SOI(kl, options={'disp': False}).minimize(s)
q_mu, q_std = s.run(q.mean).squeeze(), np.diag(s.run(q.var.mat)) ** .5
plt.figure()
plt.plot(x.squeeze(), p_mu, label='p', c='tab:green')
plt.plot(x.squeeze(), p_mu + 2 * p_std, ls='--', c='tab:green')
plt.plot(x.squeeze(), p_mu - 2 * p_std, ls='--', c='tab:green')
plt.plot(x.squeeze(), q_mu, label='q (KL)', c='tab:orange')
plt.plot(x.squeeze(), q_mu + 2 * q_std, ls='--', c='tab:orange')
plt.plot(x.squeeze(), q_mu - 2 * q_std, ls='--', c='tab:orange')
plt.legend()

print('KL after KL minimisation: {}'.format(s.run(kl)))
print('W2 after KL minimisation: {}'.format(s.run(w2)))
print('Scale after KL minimisation: {}'.format(s.run(scale)))
print('Var after KL minimisation: {}\n'.format(s.run(var)))

s.run(tf.variables_initializer([q_L, q_mu2, log_scale, log_var]))

SOI(w2, options={'disp': False}).minimize(s)
q_mu, q_std = s.run(q.mean).squeeze(), np.diag(s.run(q.var.mat)) ** .5
plt.figure()
plt.plot(x.squeeze(), p_mu, label='p', c='tab:green')
plt.plot(x.squeeze(), p_mu + 2 * p_std, ls='--', c='tab:green')
plt.plot(x.squeeze(), p_mu - 2 * p_std, ls='--', c='tab:green')
plt.plot(x.squeeze(), q_mu, label='q (W2)', c='tab:orange')
plt.plot(x.squeeze(), q_mu + 2 * q_std, ls='--', c='tab:orange')
plt.plot(x.squeeze(), q_mu - 2 * q_std, ls='--', c='tab:orange')
plt.legend()

print('KL after W2 minimisation: {}'.format(s.run(kl)))
print('W2 after W2 minimisation: {}'.format(s.run(w2)))
print('Scale after W2 minimisation: {}'.format(s.run(scale)))
print('Var after W2 minimisation: {}\n'.format(s.run(var)))

plt.show()
