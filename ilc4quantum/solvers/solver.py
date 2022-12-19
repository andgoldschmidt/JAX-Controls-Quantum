import functools
import jax
import jax.numpy as jnp


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
    Sovler is a sequential solver for optimal control problems defined by taking linear-quadratic approximatons along
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
        'linear_model_fn': linearize_model,
        'cost_fn': cost_fn,
        'approx_cost': quadraticize_cost,
        'u_sat': u_sat,
        'du_sat': du_sat
    }

    solver_iteration = jax.tree_util.Partial(methods.get(name), **solver_kwargs)

    def rollout_decorator(carry, scan):
        (next_tx, next_tu), step = solver_iteration(carry, scan)
        rollout_next_tx = solve_ivp(x_init, next_tu, rollout_fn)
        return (rollout_next_tx, next_tu), step

    (tx, tu), steps = jax.lax.scan(rollout_decorator, (tx_guess, tu_guess), None, length=max_iter)
    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1), steps


def iteration_solve_ivp(current_x, current_u, model_fn):
    """ Input: carry and scan. Return: carry and save. """
    return model_fn(current_x, current_u), jnp.hstack((current_x, current_u))


@functools.partial(jax.jit, static_argnames=["model_fn"])
def solve_ivp(x_init, tu_carry, model_fn):
    n_state = len(x_init)
    x_end, tz_soln = jax.lax.scan(jax.tree_util.Partial(iteration_solve_ivp, model_fn=model_fn), x_init, tu_carry)
    return jnp.concatenate((tz_soln[:, :n_state], x_end[None]), axis=0)
