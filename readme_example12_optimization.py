import matplotlib.pyplot as plt
from wbml.plot import tweak
import torch
from stheno import EQ, GP
from varz.torch import Vars, minimise_l_bfgs_b
from varz.spec import parametrised, Positive
import lab.torch as B

B.set_random_seed(42)


# Sample a true, underlying function and observations with known noise.
x_obs = B.linspace(0, 2, 50)
x = B.linspace(0, 2, 101)
true_func = lambda x: B.sin(5 * x)
f_true = true_func(x_obs)
true_noise_var = 0.01
y_obs = f_true + (true_noise_var ** 0.5) * B.randn(x_obs.shape[0])


# Construct a model with learnable parameters.
def model(vs):
    # Varz handles positivity (and other) constraints.
    kernel = vs.positive(name="variance") * EQ().stretch(vs.positive(name="ls"))
    noise_var = vs.positive(name="noise_var")
    return GP(kernel), noise_var


# A more convenient way of defining above model
# @parametrised
# def model(vs, ls: Positive, variance: Positive, noise_var: Positive):
#     """Constuct a model with learnable parameters."""
#     kernel = variance * EQ().stretch(ls)
#     return GP(kernel), noise_var


# Define an objective function.
def objective(vs):
    f, noise_var = model(vs)
    return -f(x_obs, noise_var).logpdf(y_obs)


# Plotting function
def plot_model(vs, title_prefix):
    f, noise_var = model(vs)
    f_post = f | (f(x_obs, noise_var), y_obs)
    mean, lower, upper = f_post(x, noise_var).marginal_credible_bounds()

    plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    plt.plot(x, true_func(x), label="True", style="test")
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    plt.ylim(-1.5, 1.5)
    plt.title(
        title_prefix
        + "\nlength scale = {:.2f}\nvariance = {:.2f}\nnoise variance = {:.2f}".format(
            vs["ls"], vs["variance"], noise_var
        )
    )
    tweak()


# Visualize initial fit
vs = Vars(torch.float64)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_model(vs, "Initial fit:")

# Perform optimization.
minimise_l_bfgs_b(objective, vs)


# Visualize final fit
plt.subplot(1, 2, 2)
plot_model(vs, "Final fit:")
plt.savefig("readme_example12_optimization.png")
plt.show()
