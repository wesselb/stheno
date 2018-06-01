import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Kronecker

# Define points to predict at.
x = np.linspace(0, 10, 100)[:, None]
x_obs = np.linspace(0, 7, 10)[:, None]

# Construct a prior.
f = GP(EQ())  # Latent function.
e = GP(0.1 * Kronecker())  # Noise.
y = f + e

# Sample a true, underlying function.
f_true = f(x).sample()

# Condition the model on the true function and sample observations.
y_obs = y.condition(f @ x, f_true)(x_obs).sample()
y.revert_prior()

# Now condition on the observations to make predictions.
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
