import jax.numpy as jnp
from tqdm import tqdm

from .ilqr import ilqr


def ilc(
        x_init,
        xs_guess,
        us_guess,
        model_fn,
        linear_model_fn,
        true_fn,
        cost_fn,
        approx_cost,
        schedule,
        u_sat=jnp.inf,
        du_sat=jnp.inf,
        algorithm='iLQR',
        warm_start=False):
    """

    :param x_init:
    :param xs_guess:
    :param us_guess:
    :param model_fn:
    :param linear_model_fn:
    :param true_fn:
    :param cost_fn:
    :param approx_cost:
    :param schedule:
    :param u_sat:
    :param du_sat:
    :param algorithm:
    :param warm_start:
    :return:
    """
