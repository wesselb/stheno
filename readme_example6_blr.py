import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, Delta, model

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
    model.sample(slope @ 0, intercept @ 0, f @ x, y @ x_obs)

# Condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)
mean_slope, mean_intercept = slope(0).mean, intercept(0).mean

print('true slope', true_slope)
print('predicted slope', mean_slope)
print('true intercept', true_intercept)
print('predicted intercept', mean_intercept)

# Plot result.
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
