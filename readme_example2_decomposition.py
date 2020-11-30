import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ, RQ, Linear, Delta, Exp, Obs, B

B.epsilon = 1e-10

# Define points to predict at.
x = B.linspace(0, 10, 200)
x_obs = B.linspace(0, 7, 50)

# Construct a latent function consisting of four different components.
prior = Measure()
f_smooth = GP(EQ(), measure=prior)
f_wiggly = GP(RQ(1e-1).stretch(0.5), measure=prior)
f_periodic = GP(EQ().periodic(1.0), measure=prior)
f_linear = GP(Linear(), measure=prior)

f = f_smooth + f_wiggly + f_periodic + 0.2 * f_linear

# Let the observation noise consist of a bit of exponential noise.
e_indep = GP(Delta(), measure=prior)
e_exp = GP(Exp(), measure=prior)

e = e_indep + 0.3 * e_exp

# Sum the latent function and observation noise to get a model for the observations.
y = f + 0.5 * e

# Sample a true, underlying function and observations.
(
    f_true_smooth,
    f_true_wiggly,
    f_true_periodic,
    f_true_linear,
    f_true,
    y_obs,
) = prior.sample(f_smooth(x), f_wiggly(x), f_periodic(x), f_linear(x), f(x), y(x_obs))

# Now condition on the observations and make predictions for the latent function and
# its various components.
post = prior | (y(x_obs), y_obs)

pred_smooth = post(f_smooth(x)).marginals()
pred_wiggly = post(f_wiggly(x)).marginals()
pred_periodic = post(f_periodic(x)).marginals()
pred_linear = post(f_linear(x)).marginals()
pred_f = post(f(x)).marginals()


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label="True", style="test")
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    mean, lower, upper = pred
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    tweak()


plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.title("Prediction")
plot_prediction(x, f_true, pred_f, x_obs, y_obs)

plt.subplot(3, 2, 3)
plt.title("Smooth Component")
plot_prediction(x, f_true_smooth, pred_smooth)

plt.subplot(3, 2, 4)
plt.title("Wiggly Component")
plot_prediction(x, f_true_wiggly, pred_wiggly)

plt.subplot(3, 2, 5)
plt.title("Periodic Component")
plot_prediction(x, f_true_periodic, pred_periodic)

plt.subplot(3, 2, 6)
plt.title("Linear Component")
plot_prediction(x, f_true_linear, pred_linear)

plt.savefig("readme_example2_decomposition.png")
plt.show()
