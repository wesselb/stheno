import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, EQ, Delta, Obs

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 10, 10)

# Construct the model.
f = 0.7 * GP(EQ()).stretch(1.5)
e = 0.2 * GP(Delta())

# Construct derivatives via finite differences.
df = f.diff_approx(1)
ddf = f.diff_approx(2)
dddf = f.diff_approx(3) + e

# Fix the integration constants.
f, df, ddf, dddf = (f, df, ddf, dddf) | Obs((f(0), 1), (df(0), 0), (ddf(0), -1))

# Sample observations.
y_obs = np.sin(x_obs) + 0.2 * np.random.randn(*x_obs.shape)

# Condition on the observations to make predictions.
f, df, ddf, dddf = (f, df, ddf, dddf) | Obs(dddf(x_obs), y_obs)

# And make predictions.
pred_iiif = f(x).marginals()
pred_iif = df(x).marginals()
pred_if = ddf(x).marginals()
pred_f = dddf(x).marginals()


# Plot result.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title('Function')
plot_prediction(x, np.sin(x), pred_f, x_obs=x_obs, y_obs=y_obs)
plt.legend()

plt.subplot(2, 2, 2)
plt.title('Integral of Function')
plot_prediction(x, -np.cos(x), pred_if)
plt.legend()

plt.subplot(2, 2, 3)
plt.title('Second Integral of Function')
plot_prediction(x, -np.sin(x), pred_iif)
plt.legend()

plt.subplot(2, 2, 4)
plt.title('Third Integral of Function')
plot_prediction(x, np.cos(x), pred_iiif)
plt.legend()

plt.show()
