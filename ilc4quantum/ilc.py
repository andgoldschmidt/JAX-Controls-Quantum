import jax
import jax.numpy as jnp
from tqdm import tqdm

from .solvers.ilqr import ilqr
from .solvers.sqp import sqp


def ilc(
        x_init,
        tu_guess,
        model_fn,
        linear_model_fn,
        true_fn,
        cost_fn,
        approx_cost,
        max_iter,
        schedule,
        u_sat=jnp.inf,
        du_sat=jnp.inf,
        solver='iLQR'):
    """
    Iterative learning control (ILC) solves an optimal control problem using a standard control solver for planning. It assumes
    rollouts of the systems are accessible as a black box. In ILC, data from the rollouts are used to iteratively compute model
    linearizations and cost quadratizations.

    :param x_init: The inital state.
    :param tu_guess: The initial control (usually an optimal control assuming the model is true).
    :param model_fn: Model dynamics. Mapping z[t] -> z[t+1]
    :param linear_model_fn: Linearized model dynamics. Mapping z[t] -> A[t].
    :param true_fn: True model dynamics (rollout access only). Mapping z[t] -> z[t+1].
    :param cost_fn: Cost function (no terminal state cost). Mapping z[0:H-1] -> Reals.
    :param approx_cost: Linearized cost function. Mapping z[0:H-1] -> Q[0:H-1], j[0:H-1].
    :param max_iter: The number of learning control iterations.
    :param schedule: The maximum number of iterations for each solver call.
    :param u_sat: Control saturation.
    :param du_sat: Slew rate.
    :param solver: Either 'iLQR' or 'SQP'. Solver type for the optimal control problem.
    :return: The optimal state of shape (horizon + 1, state), the optimal control of shape (horizon, state), the replay buffer
    """
    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess
    n_state = len(x_init)
    max_iter = 1

    # Select solver.
    if solver == 'iLQR':
        controller = ilqr
    elif solver == 'SQP':
        controller = sqp
    else:
        raise ValueError(f'Invalid solver {solver}. Value must be one of iLQR, SQP.')

    # Integrate true model along x_init using tu_guess
    rollout = jax.tree_util.Partial(solve_ivp, model_fn=true_fn)

    # Find optimal control
    # TODO: Write as a jax.lax.scan.
    replay_buffer = [None] * max_iter
    for i in tqdm(range(max_iter), total=max_iter):
        # Compute rollout
        tx_guess = rollout(x_init, tu_guess)

        # Solve optimal control
        tz_opt = controller(
            x_init,
            tx_guess,
            tu_guess,
            model_fn,
            linear_model_fn,
            cost_fn,
            approx_cost,
            u_sat,
            du_sat,
            schedule[i],
        )

        # Buffer
        # TODO: Cost function has no final state cost.
        replay_buffer[i] = {
            'state': tz_opt[:, :n_state],
            'control': tz_opt[:-1, -n_ctrl:],
            'cost': cost_fn(tz_opt[:-1])
        }

        # Update control
        tu_guess = tz_opt[:-1, -n_ctrl:]


    return tz_opt[:, :n_state], tz_opt[:-1, -n_ctrl:], replay_buffer


def iteration_solve_ivp(current_x, current_u, model_fn):
    """ Input: carry and scan. Return: carry and save. """
    current_z = jnp.hstack((current_x, current_u))
    return model_fn(current_z), current_z


def solve_ivp(x_init, tu_carry, model_fn):
    n_state = len(x_init)
    x_end, tz_soln = jax.lax.scan(jax.tree_util.Partial(iteration_solve_ivp, model_fn=model_fn), x_init, tu_carry)
    return jnp.concatenate((tz_soln[:, :n_state], x_end[None]), axis=0)
