import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import vars64 as vs

from stheno.tf import GP, EQ, Kronecker, model

s = tf.Session()

# Define points to predict at.
x = np.linspace(0, 5, 100)[:, None]
x_obs = np.linspace(0, 3, 20)[:, None]

# Construct the model.
u = GP(vs.pos(.5) * EQ().stretch(vs.pos(1.)))
e = GP(vs.pos(.5) * Kronecker())
alpha = vs.pos(1.2)
vs.init(s)

f = u + (lambda x: x ** alpha)
y = f + e

# Sample a true, underlying function and observations.
f_true = x ** 1.8
y_obs = s.run(y.condition(f @ x, f_true)(x_obs).sample())
model.revert_prior()

# Learn.
lml = y(x_obs).log_pdf(y_obs)
SOI(-lml).minimize(s)

# Print the learned parameters.
print('alpha', s.run(alpha))
print('noise', s.run(e.var))
print('u scale', s.run(u.length_scale))
print('u variance', s.run(u.var))

# Condition on the observations to make predictions.
mean, lower, upper = s.run(f.condition(y @ x_obs, y_obs).predict(x))

# Plot result.
x, f_true, x_obs, y_obs = map(np.squeeze, (x, f_true, x_obs, y_obs))
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
