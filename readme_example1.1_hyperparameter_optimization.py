from stheno import EQ, GP
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b
import lab.tensorflow as B

# Sample a true, underlying function and observations with known noise.
x_obs = B.linspace(0, 2, 50)
ls_true = 0.2 # Lengthscale                   
variance_true = 0.5
f_fixed = GP(variance_true * EQ().stretch(ls_true))
noise_true = 0.01
f_true, y_obs = f_fixed.measure.sample(f_fixed(x_obs), f_fixed(x_obs, noise_true))

# Construct a model with learnable parameters.
def model(vs):
    # Varz handles positivity (and other) constraints.
    kernel = variance_true * EQ().stretch(vs.positive(name="ls"))
    noise = noise_true
    return GP(kernel), noise

# Define an objective function.
def objective(vs):
    f, noise = model(vs)
    return -f(x_obs, noise).logpdf(y_obs)

# Perform optimisation and print the learned parameters.
vs = Vars(tf.float64)
minimise_l_bfgs_b(objective, vs, trace=True)
vs.print()