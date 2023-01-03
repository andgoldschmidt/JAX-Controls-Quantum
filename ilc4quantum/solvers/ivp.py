import functools
import jax
import jax.numpy as jnp


def iteration_solve_ivp(current_x, current_u, model_fn):
    """ Input: carry and scan. Return: carry and save. """
    return model_fn(current_x, current_u), jnp.hstack((current_x, current_u))


@functools.partial(jax.jit, static_argnames=["model_fn"])
def solve_ivp(x_init, tu_carry, model_fn):
    n_state = len(x_init)
    x_end, tz_soln = jax.lax.scan(jax.tree_util.Partial(iteration_solve_ivp, model_fn=model_fn), x_init, tu_carry)
    return jnp.concatenate((tz_soln[:, :n_state], x_end[None]), axis=0)