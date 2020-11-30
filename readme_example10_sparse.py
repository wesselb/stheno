import matplotlib.pyplot as plt
import wbml.out as out
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ, Delta, SparseObs

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 7, 50_000)
x_ind = B.linspace(0, 10, 20)

# Construct a prior.
prior = Measure()
f = GP(EQ().periodic(2 * B.pi), measure=prior)  # Latent function.
e = GP(Delta(), measure=prior)  # Noise.
y = f + 0.5 * e

# Sample a true, underlying function and observations.
f_true = B.sin(x)
y_obs = B.sin(x_obs) + 0.5 * B.randn(*x_obs.shape)

# Now condition on the observations to make predictions.
obs = SparseObs(
    f(x_ind),  # Inducing points.
    0.5 * e,  # Noise process.
    # Observations _without_ the noise process added on.
    f(x_obs),
    y_obs,
)
out.kv("ELBO", obs.elbo(prior))
post = prior | obs
mean, lower, upper = post(f(x)).marginals()

# Plot result.
plt.plot(x, f_true, label="True", style="test")
plt.scatter(
    x_obs, y_obs, label="Observations", style="train", c="tab:green", alpha=0.35
)
plt.scatter(
    x_ind,
    obs.mu(prior)[:, 0],
    label="Inducing Points",
    style="train",
    s=20,
)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example10_sparse.png")
plt.show()
