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

# Sample a true slope and intercept.
true_slope = slope(0).sample()
true_intercept = intercept.condition(slope @ 0, true_slope)(0).sample()

# Sample a true, underlying function and observations.
f_true = f.condition(intercept @ x, true_intercept)(x).sample()
y_obs = y.condition(f @ x, f_true)(x_obs).sample()
model.revert_prior()

# Condition on the observations to make predictions.
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)
mean_slope, mean_intercept = slope(0).mean, intercept(0).mean

print('true slope', true_slope)
print('predicted slope', mean_slope)
print('true intercept', true_intercept)
print('predicted intercept', mean_intercept)

# Plot result.
x, f_true, x_obs, y_obs = map(np.squeeze, (x, f_true, x_obs, y_obs))
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()
plt.show()
