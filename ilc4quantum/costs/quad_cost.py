import jax
import jax.numpy as jnp


# TODO: general quadratic form (conjugate transpose).
# TODO: costs should include final state cost
#  E.g. zs should have size n + 1, and assume zero control for n + 1,
#       instead of size n with a (presumed) zero state for n + 1. This
#       also requires adjustments around zs, Hs, and Js within mpc code.

# vmap matrix multiplication
matmul_vmap = jax.vmap(jnp.matmul)


def quad_cost_fn(tz, tz_ref, tH_cost, tj_cost, discount=1):
    # Discount factors (time axis: 0)
    t_dfs = discount ** jnp.arange(len(tz))

    # Quadratic cost
    dzs_ref = tz - tz_ref
    zHz = matmul_vmap(dzs_ref, matmul_vmap(t_dfs[:, None, None] * tH_cost, dzs_ref))
    jz = matmul_vmap(t_dfs[:, None] * tj_cost, dzs_ref)
    return jnp.sum(zHz / 2 + jz, axis=0)


def quadraticize_quad_cost(tz_guess, tz_ref, tH_cost, tj_cost, discount=1):
    # Broadcast discount factors (time axis: 0)
    t_dfs = discount ** jnp.arange(len(tz_guess))

    # Expand around guess trajectory instead of reference
    # Assumes: Hs_cost is symmetric
    tj_new = matmul_vmap(t_dfs[:, None, None] * tH_cost, tz_guess - tz_ref) + t_dfs[:, None] * tj_cost
    tH_new = t_dfs[:, None, None] * tH_cost
    return tH_new, tj_new
