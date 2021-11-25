import matplotlib.pyplot as plt
from wbml.plot import tweak
import torch
from stheno import EQ, GP
import lab.torch  # torch context necessary for stheno

# Enable to reproduce the same results repeatedly.
# torch.manual_seed(0)


# Sample a true, underlying function and observations with known noise.
x_obs = torch.linspace(0, 2, 50).double()
x = torch.linspace(0, 2, 101).double()
true_func = lambda x: torch.sin(5 * x)
f_true = true_func(x_obs)
true_noise_var = 0.01
y_obs = f_true + (true_noise_var ** 0.5) * torch.randn(x_obs.shape[0]).double()


# Define a positivity constraint.
def positive(variable):
    return torch.exp(variable)


# Construct a model with learnable parameters (positive).
def model(scale, variance, noise_var):
    kernel = positive(variance) * EQ().stretch(positive(scale))
    return GP(kernel), positive(noise_var)


# Define an objective function.
def objective(scale, variance, noise_var):
    f, pos_noise_var = model(scale, variance, noise_var)
    return -f(x_obs, pos_noise_var).logpdf(y_obs)


# Plotting function
def plot_model(scale, variance, noise_var, title_prefix):
    f, pos_noise_var = model(scale, variance, noise_var)
    f_post = f | (f(x_obs, pos_noise_var), y_obs)
    mean, lower, upper = f_post(x, pos_noise_var).marginal_credible_bounds()

    plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    plt.plot(x, true_func(x), label="True", style="test")
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    plt.ylim(-1.5, 1.5)
    plt.title(
        title_prefix
        + "\nlength scale = {:.2f}\nvariance = {:.2f}\nnoise variance = {:.2f}".format(
            f.kernel.factor(1).stretches[0].item(),
            f.kernel.factor(0).item(),
            pos_noise_var.item(),
        )
    )
    tweak()


# Create trainable torch variables
scale = torch.empty((), dtype=torch.float64, requires_grad=True)
variance = torch.empty((), dtype=torch.float64, requires_grad=True)
noise_var = torch.empty((), dtype=torch.float64, requires_grad=True)


# Initialise the variables
for param in [scale, variance, noise_var]:
    # Log transform as an inverse transformation to positive (exp) transform
    param.data.fill_(torch.log(torch.rand(())))


# Visualize initial fit
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plot_model(scale, variance, noise_var, "Initial fit:")


# Define an optimizer
optimizer = torch.optim.Adam([scale, variance, noise_var], lr=0.1)


# Perform optimization.
n_iters = 50
for _ in range(n_iters):
    optimizer.zero_grad()
    loss = objective(scale, variance, noise_var)
    loss.backward()
    optimizer.step()


# Visualize final fit
plt.subplot(1, 2, 2)
plot_model(scale, variance, noise_var, "Final fit:")
plt.savefig("readme_example13_optimization_torch.png")
plt.show()
