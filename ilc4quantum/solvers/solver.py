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


    :param name: Name of solver to use.
    :param x_init: Initial state.
    :param u_init: Inital control.
    :param tx_guess: Shape is (time, state)
    :param tu_guess: Shape is (time, control)
    :param model_fn: Model dynamics. Mapping z[t] -> z[t+1]
    :param linearize_model: Linearized model dynamics. Mapping z[t] -> A[t]
    :param cost_fn: Cost function (no terminal state cost). Mapping z[0:H-1] -> Reals
    :param quadraticize_cost: Quadraticized cost function. Mapping z[0:H-1] -> Q[0:H-1], j[0:H-1]
    :param u_sat: Control saturation.
    :param du_sat: Slew rate.
    :param max_iter: Maximumum number of iLQR steps.
    :return: The optimal trajectory, z[0:H]. A zero control is appended alongside the terminal state, z[H].
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
    (tx, tu), steps = jax.lax.scan(solver_iteration, (tx_guess, tu_guess), None, length=max_iter)
    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1)
