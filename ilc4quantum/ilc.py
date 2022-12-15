import jax.numpy as jnp


def ilc(
        x_init,
        tx_guess,
        tu_guess,
        model_fn,
        linear_model_fn,
        true_fn,
        cost_fn,
        approx_cost,
        schedule,
        u_sat=jnp.inf,
        du_sat=jnp.inf,
        solver='iLQR'):
    """

    :param x_init:
    :param tx_guess:
    :param tu_guess:
    :param model_fn:
    :param linear_model_fn:
    :param true_fn:
    :param cost_fn:
    :param approx_cost:
    :param schedule:
    :param u_sat:
    :param du_sat:
    :param solver:
    :return:
    """
