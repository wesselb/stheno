import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Delta, model

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 10, 10)

# Construct the model.
f = GP(.5) * (lambda x: x) + GP(1)
e = 0.3 * GP(Delta())
y = f + e

# Sample a true, underlying function and observations.
f_true = f(x).sample()
y_obs = y.condition(f @ x, f_true)(x_obs).sample()
model.revert_prior()

# Condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)

# Plot result.
x, f_true, x_obs, y_obs = map(np.squeeze, (x, f_true, x_obs, y_obs))
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
