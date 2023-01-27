import functools
from itertools import combinations
import jax
import jax.numpy as jnp
import numpy as np

from .ivp import solve_ivp
from ..dynamics import create_power_list, dmd_model, lift_model, fit_model_split
from ..costs import quad_cost_fn, quadraticize_quad_cost


methods = {}


def register(name=""):
    def inner(f):
        n = name or f.__name__
        methods[n] = f
        return f
    return inner


solver_static_args = ['name', 'max_iter', 'model_fn', 'linearize_model', 'cost_fn', 'quadraticize_cost']


@functools.partial(jax.jit, static_argnames=solver_static_args)
def solver(
        name,
        x_init,
        u_init,
        tx_guess,
        tu_guess,
        rollout_fn,
        model_fn,
        linearize_model,
        cost_fn,
        quadraticize_cost,
        u_sat,
        du_sat,
        max_iter):
    """
    Solver is a sequential solver for optimal control problems defined by taking linear-quadratic approximatons along
    guess trajectories. The key interface of a solver is the implementation of a solver_iteration. This is any function
    with signature `solver_iteration(carry, scan, **kwargs)` that takes a `carry=(tx_guess, tu_guess)` and runs a scan
    over the iteration number. The return value of a solver_iteration is an udpated value for the carry,
    `carry=(xs_guess_next, us_guess_next)`, and the step size taken. For more information, consult the documentation
    for `jax.lax.scan`.

    Notes:
        1.  It is important to have variables like max_iter last (else: "non-hashable static args...")
        2.  The `rollout_fn` is used to evaluate the forward iteration (to simulate using an iterative learning
            controller) after the solver has completed the current iteration. No knowledge of the underlying rollout
            model is accessible; within the solver (e.g. linearization, line search, feedback) `model_fn` is used.
            Rollout_fn must call as f(x,u) and not as f(z) because unpacking z results in erros, either for
            `Non-hashable static arguments...` or `Array slice indices must have static start/stop/step...`
            NOTE: iLQR requires `rollout_fn` is available because the LQR solution has a feedback term at each control step.
        3.  TODO: this can probably be simplified to a normal for loop to avoid finicky coding in the first iteration.

    :param name: Name of solver to use.
    :param x_init: Initial state.
    :param u_init: Inital control.
    :param tx_guess: Shape is (time, state)
    :param tu_guess: Shape is (time, control),
    :param rollout_fn: True dynamics. Only called in a forward pass. Mapping (x[t], u[t]) -> x[t+1].
    :param model_fn: Model dynamics. Mapping (x[t], u[t]) -> x[t+1]
    :param linearize_model: Linearized model dynamics. Mapping z[t] -> A[t]
    :param cost_fn: Cost function (no terminal state cost). Mapping z[0:H-1] -> Reals
    :param quadraticize_cost: Quadraticized cost function. Mapping z[0:H-1] -> Q[0:H-1], j[0:H-1]
    :param u_sat: Control saturation.
    :param du_sat: Slew rate.
    :param max_iter: Maximumum number of iLQR steps.
    :return: The optimal trajectory, z[0:H], and the optimization step sizes taken. For the trajecotry, a zero control
     is appended alongside the terminal state, z[H].
    """
    solver_kwargs = {
        'x_init': x_init,
        'u_init': u_init,
        'model_fn': model_fn,
        'rollout_fn': rollout_fn,
        'linear_model_fn': linearize_model,
        'cost_fn': cost_fn,
        'approx_cost': quadraticize_cost,
        'u_sat': u_sat,
        'du_sat': du_sat,
    }

    solver_iteration = jax.tree_util.Partial(methods.get(name), **solver_kwargs)

    def rollout_decorator(carry, scan):
        """ Can be used to add noise or other challenges to readout data. """
        (_, next_tu), step = solver_iteration(carry, scan)
        rollout_next_tx = solve_ivp(x_init, next_tu, rollout_fn)
        return (rollout_next_tx, next_tu), step

    (tx, tu), steps = jax.lax.scan(rollout_decorator, (tx_guess, tu_guess), None, length=max_iter)
    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1), steps


def model_discovery_solver(
        name,
        x_init,
        u_init,
        tx_guess,
        tu_guess,
        tx_ref,
        tu_ref,
        rollout_fn,
        initial_fn,
        cost_matrix,
        u_sat,
        du_sat,
        dmd_lift,
        dmd_rank,
        max_iter):
    """
    Add model discovery to the outer loop.
    """
    dim_x = len(x_init)
    dim_t, dim_u = tu_guess.shape

    # Fix the model discovery order
    control_powers = create_power_list(dmd_lift, dim_u)

    # Construct costs based on initial references and external cost matrix
    cost_kwargs = {
        'tz_ref': jnp.concatenate((tx_ref[:-1], tu_ref), axis=1),
        'tH_cost': jnp.repeat(cost_matrix[None], dim_t, axis=0),
        'tj_cost': jnp.zeros((dim_t, dim_x + dim_u)),
        'discount': 1.
    }
    cost_fn = functools.partial(quad_cost_fn, **cost_kwargs)
    quadraticize_cost = functools.partial(quadraticize_quad_cost, **cost_kwargs)

    solver_kwargs = {
        'x_init': x_init,
        'u_init': u_init,
        'rollout_fn': rollout_fn,
        'cost_fn': cost_fn,
        'approx_cost': quadraticize_cost,
        'u_sat': u_sat,
        'du_sat': du_sat,
    }
    # TODO: How can we make this fast with JIT?
    solver_iteration = jax.tree_util.Partial(methods.get(name), **solver_kwargs)

    # Initialize loop variables
    model_fn = initial_fn
    tx = tx_guess
    tu = tu_guess
    steps = [None] * max_iter
    for i in range(max_iter):
        # Apply ILC iteration
        dmd_kwargs = {
            'model_fn': model_fn,
            'linear_model_fn': jax.jacfwd(lift_model(model_fn, dim_x, dim_u)),
            'cost_fn': cost_fn,
            'approx_cost': quadraticize_cost,
        }
        dmd_solver_iteration = jax.tree_util.Partial(solver_iteration, **dmd_kwargs)
        (_, tu), steps[i] = dmd_solver_iteration((tx, tu), i)
        tx = solve_ivp(x_init, tu, rollout_fn)

        # Update costs
        cost_kwargs["tz_ref"] = jnp.concatenate((tx_ref[:-1], tu), axis=1)
        cost_fn = functools.partial(quad_cost_fn, **cost_kwargs)
        quadraticize_cost = functools.partial(quadraticize_quad_cost, **cost_kwargs)

        # Update model
        if i == 0:
            tx2_buffer = tx[1:]
            tx1_buffer = tx[:-1]
            tu_buffer = tu
        else:
            tx2_buffer = jnp.concatenate([tx2_buffer, tx[1:]], axis=0)
            tx1_buffer = jnp.concatenate([tx1_buffer, tx[:-1]], axis=0)
            tu_buffer = jnp.concatenate([tu_buffer, tu], axis=0)
        model_fn = jax.tree_util.Partial(
            dmd_model, 
            A_op=fit_model_split(tx2_buffer, tx1_buffer, tu_buffer, control_powers, dmd_rank),
            powers=control_powers
        )

    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1), steps
