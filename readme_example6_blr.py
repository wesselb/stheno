import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, Delta, model, Obs, dense

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 10, 10)

# Construct the model.
slope = GP(1)
intercept = GP(5)
f = slope * (lambda x: x) + intercept

e = 0.2 * GP(Delta())  # Noise model

y = f + e  # Observation model

# Sample a slope, intercept, underlying function, and observations.
true_slope, true_intercept, f_true, y_obs = \
    model.sample(slope(0), intercept(0), f(x), y(x_obs))

# Condition on the observations to make predictions.
slope, intercept, f = (slope, intercept, f) | Obs(y(x_obs), y_obs)
mean, lower, upper = f.predict(x)

print('true slope', true_slope)
print('predicted slope', slope(0).mean)
print('true intercept', true_intercept)
print('predicted intercept', intercept(0).mean)

# Plot result.
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
