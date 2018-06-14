import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, model, EQ, RQ, Linear, Delta, Exp

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs = np.linspace(0, 7, 50)

# Construct a latent function consisting of four different components.
f_smooth = GP(EQ())
f_wiggly = GP(RQ(1e-1).stretch(.5))
f_periodic = GP(EQ().periodic(1.))
f_linear = GP(Linear())

f = f_smooth + f_wiggly + f_periodic + .2 * f_linear

# Let the observation noise consist of a bit of exponential noise.
e_indep = GP(Delta())
e_exp = GP(Exp())

e = e_indep + .3 * e_exp

# Sum the latent function and observation noise to get a model for the
# observations.
y = f + .5 * e

# Sample a true, underlying function and observations.
f_true_smooth, f_true_wiggly, f_true_periodic, f_true_linear, f_true, y_obs = \
    model.sample(f_smooth @ x,
                 f_wiggly @ x,
                 f_periodic @ x,
                 f_linear @ x,
                 f @ x,
                 y @ x_obs)

# Now condition on the observations and make predictions for the latent
# function and its various components.
model.condition(y @ x_obs, y_obs)

pred_smooth = f_smooth.predict(x)
pred_wiggly = f_wiggly.predict(x)
pred_periodic = f_periodic.predict(x)
pred_linear = f_linear.predict(x)
pred_f = f.predict(x)


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x.squeeze(), f.squeeze(), label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs.squeeze(), y_obs.squeeze(),
                    label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x.squeeze(), mean, label='Prediction', c='tab:green')
    plt.plot(x.squeeze(), lower, ls='--', c='tab:green')
    plt.plot(x.squeeze(), upper, ls='--', c='tab:green')
    plt.legend()


plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.title('Prediction')
plot_prediction(x, f_true, pred_f, x_obs, y_obs)

plt.subplot(3, 2, 3)
plt.title('Smooth Component')
plot_prediction(x, f_true_smooth, pred_smooth)

plt.subplot(3, 2, 4)
plt.title('Wiggly Component')
plot_prediction(x, f_true_wiggly, pred_wiggly)

plt.subplot(3, 2, 5)
plt.title('Periodic Component')
plot_prediction(x, f_true_periodic, pred_periodic)

plt.subplot(3, 2, 6)
plt.title('Linear Component')
plot_prediction(x, f_true_linear, pred_linear)

plt.show()
