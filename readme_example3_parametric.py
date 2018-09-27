import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import vars64 as vs

from stheno.tf import GP, EQ, Delta

s = tf.Session()

# Define points to predict at.
x = np.linspace(0, 5, 100)
x_obs = np.linspace(0, 3, 20)

# Construct the model.
u = GP(vs.pos(.5) * EQ().stretch(vs.pos(1.)))
e = GP(vs.pos(.5) * Delta())
alpha = vs.pos(1.2)
vs.init(s)

f = u + (lambda x: x ** alpha)
y = f + e

# Sample a true, underlying function and observations.
f_true = x ** 1.8
y_obs = s.run((y | (f(x), f_true))(x_obs).sample())

# Learn.
lml = y(x_obs).logpdf(y_obs)
SOI(-lml).minimize(s)

# Print the learned parameters.
print('alpha', s.run(alpha))
print('prior', y.display(s.run))

# Condition on the observations to make predictions.
mean, lower, upper = s.run((f | (y(x_obs), y_obs)).predict(x))

# Plot result.
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
