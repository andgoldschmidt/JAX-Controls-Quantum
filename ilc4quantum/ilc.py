import jax
import jax.numpy as jnp
from tqdm import tqdm

from .solvers import solver, solve_ivp


def ilc(
        x_init,
        tu_guess,
        model_fn,
        linearize_model,
        true_fn,
        cost_fn,
        quadraticize_cost,
        max_iter,
        schedule,
        u_sat=jnp.inf,
        du_sat=jnp.inf,
        solver_name='iLQR'):
    """
    TODO: Construct a top-level ILC and MPC framework for comparison of the two approaches.
    NOTE: For batch processes: MPC can be used as the first pass, but ILC should otherwise be favored.
    NOTE: For tracking: Model discovery and re-planning is important if there are infeasible trajectories due to static
          model parameters.

    Iterative learning control (ILC) solves an optimal control problem using a standard control solver for planning.
    It assumes rollouts of the systems are accessible as a black box. In ILC, data from the rollouts are used to
    iteratively compute model linearizations and cost quadratizations.

    :param x_init: The inital state.
    :param tu_guess: The initial control (usually an optimal control assuming the model is true).
    :param model_fn: Model dynamics. Mapping z[t] -> z[t+1]
    :param linearize_model: Linearized model dynamics. Mapping z[t] -> A[t].
    :param true_fn: True model dynamics (rollout access only). Mapping z[t] -> z[t+1].
    :param cost_fn: Cost function (no terminal state cost). Mapping z[0:H-1] -> Reals.
    :param quadraticize_cost: Linearized cost function. Mapping z[0:H-1] -> Q[0:H-1], j[0:H-1].
    :param max_iter: The number of learning control iterations.
    :param schedule: The maximum number of iterations for each solver call.
    :param u_sat: Control saturation.
    :param du_sat: Slew rate.
    :param solver_name: Either 'iLQR' or 'SQP'. Name of solver for the optimal control problem.
    :return: The optimal state of shape (horizon + 1, state), the optimal control of shape (horizon, state), the replay buffer
    """
    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess
    n_state = len(x_init)
    max_iter = 1

    # Integrate true model along x_init using tu_guess
    rollout = jax.tree_util.Partial(solve_ivp, model_fn=true_fn)

    # Find optimal control
    replay_buffer = [None] * max_iter
    for i in tqdm(range(max_iter), total=max_iter):
        # Compute rollout
        tx_guess = rollout(x_init, tu_guess)

        # Solve optimal control
        tz_opt = solver(
            solver_name,
            x_init,
            tx_guess,
            tu_guess,
            model_fn,
            linearize_model,
            cost_fn,
            quadraticize_cost,
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

