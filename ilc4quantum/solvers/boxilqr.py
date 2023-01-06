import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP, BacktrackingLineSearch

from .ilqr import backtracking_line_search, discount_fn, forward_pass, scan_forward
from .solver import register


@register("Box-iLQR")
def iteration_box_lqr(
        carry,
        iteration,
        x_init,
        u_init,
        model_fn,
        rollout_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat):
    """
    Iterative LQR with box constraints (1st order in dynamics, 2nd order in cost).


    Read more:  *  10.1109/IROS.2012.6386025
                *  10.1109/ICRA.2014.6907001

    Notes:  1.  It can sometimes be important to enable 'jax_enable_x64' for iLQR.
            2.  Slew bounds must account for the rate from the control guess.
            3.  TODO: Gradient information and step should inform convergence / exit.
            4.  There is a state feedback component to this controller. Therefore, the rollout function needs to be applied directly
                within iLQR for any ILC applications. One might alteratively consider using the reference state to provide feedback, 
                but this does not work.
    """
    tx_guess, tu_guess = carry

    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess.shape
    _, n_state = tx_guess.shape
    tz_guess = jnp.concatenate([tx_guess, jnp.vstack([tu_guess, jnp.zeros(n_ctrl)])], axis=1)

    # Accumulated slew
    slew_init = tu_guess[0] - u_init
    tu_guess_slew = jnp.vstack([slew_init[None], tu_guess[1:] - tu_guess[:-1]])

    # 1. Compute derivatives
    # -- Linear expansion of model
    tF_linear = jax.vmap(linear_model_fn)(tz_guess[:-1])
    # -- Quadratic expansion of cost about tz_guess
    tH_cost, tj_cost = approx_cost(tz_guess[:-1])

    # 2. Backward pass
    prob = BoxCDQP()
    fwd = jnp.zeros(n_ctrl)
    V_x = tj_cost[-1, :n_state]
    V_xx = tH_cost[-1, :n_state, :n_state]
    # TODO: Search mu?
    _, feeds = jax.lax.scan(jax.tree_util.Partial(scan_box_backward, mu=1e-9, u_sat=u_sat, du_sat=du_sat, prob=prob),
                            (fwd, V_x, V_xx),
                            (tu_guess, tu_guess_slew, tj_cost, tH_cost, tF_linear),
                            reverse=True)
    tu_feedforward, tux_Feedback, delta_V1s, delta_V2s = feeds

    # 3. Forward pass
    # -- Separate pass for model (linesearch) and rollout (optimization)
    model_partial = jax.tree_util.Partial(scan_forward, u_sat=u_sat, du_sat=du_sat, model_fn=model_fn)
    model_step = jax.tree_util.Partial(
        forward_pass, 
        scan_forward_step=model_partial, 
        carry_init=(x_init, u_init), 
        scan=(tz_guess[:-1], tu_feedforward, tux_Feedback))

    rollout_partial = jax.tree_util.Partial(scan_forward, u_sat=u_sat, du_sat=du_sat, model_fn=rollout_fn)
    rollout_step = jax.tree_util.Partial(
        forward_pass, 
        scan_forward_step=rollout_partial, 
        carry_init=(x_init, u_init), 
        scan=(tz_guess[:-1], tu_feedforward, tux_Feedback))

    # -- Line search parameters
    # TODO: What are the correct tz_guess indicies? prev. was tz_guess[1:]
    cost_guess = cost_fn(tz_guess[:-1])
    # wolfe_c1 = 0.0001
    # TODO: Is this dcost_lin the correct type of line search linearization?
    dcost_lin = jnp.sum(delta_V1s, axis=0)
    # wolfe_c2 = 0.9
    # dcost_quad = jnp.sum(delta_V2s, axis=0)
    step = jax.lax.while_loop(
        jax.tree_util.Partial(backtracking_line_search,
                              forward_pass_step=model_step,
                              cost_fn=cost_fn,
                              cost_guess=cost_guess,
                              dcost_lin=dcost_lin),
        discount_fn,
        1.
    )
    # # TODO: I sometimes get better results taking a full step than I do following the line search. Why?
    # step = 1.

    # -- Take step
    # NOTE: This was a test of using the reference state as the feedback (idea: x -> x_ref, eventually). It didn't work.
    # tu_step = tu_guess + step * tu_feedforward + jnp.einsum("tux,tx->tu", tux_Feedback, tx_ref[:-1] - tx_guess[:-1])
    # tu_prev = jnp.concatenate([u_init[None], tu_step[:-1]], axis=0)
    # tu_next = jax.lax.clamp(jnp.maximum(tu_prev - du_sat, -u_sat), tu_step, jnp.minimum(du_sat + tu_prev, u_sat))
    (x_end, _), tz_opt = rollout_step(step)
    tx_next = jnp.vstack([tz_opt[:, :n_state], x_end])
    tu_next = tz_opt[:, -n_ctrl:]

    return (tx_next, tu_next), (step, jnp.max(tu_feedforward, axis=0))


def scan_box_backward(carry, scan, mu, u_sat, du_sat, prob):
    next_fwd, next_V_x, next_V_xx = carry
    u_guess, u_guess_slew, J_cost, H_cost, F_linear = scan
    # Pay attention to shapes!
    n_ctrl = len(u_guess)
    n_state = len(next_V_x)

    # 1. Approximate the Bellman value function
    # TODO: Make this DDP for faster convergence (requires F_ij)
    F_x = F_linear[:, :n_state]
    F_u = F_linear[:, -n_ctrl:]
    Q_u = J_cost[-n_ctrl:] + F_u.T @ next_V_x
    Q_x = J_cost[:n_state] + F_x.T @ next_V_x
    Q_xx = H_cost[:n_state, :n_state] + jnp.linalg.multi_dot([F_x.T, next_V_xx, F_x])
    # -- Regularize with a quadratic state-cost around the previous sequence.
    Q_uu = H_cost[-n_ctrl:, -n_ctrl:] + jnp.linalg.multi_dot([F_u.T, next_V_xx + mu * jnp.identity(n_state), F_u])
    Q_ux = H_cost[-n_ctrl:, :n_state] + jnp.linalg.multi_dot([F_u.T, next_V_xx + mu * jnp.identity(n_state), F_x])

    # 2. Optimize
    # -- Control inequality constraints.
    du_min = -jnp.minimum(du_sat + u_guess_slew, u_sat + u_guess)
    du_max = jnp.minimum(du_sat - u_guess_slew, u_sat - u_guess)
    # -- Solve the box QP (warm start with prev. step) (unconstrained, fwd = -Q_uu_inv @ Q_u)
    fwd, state = prob.run(next_fwd, params_obj=(Q_uu, Q_u), params_ineq=(du_min, du_max))

    # 3. Compute Feedback
    # -- Construct a boolean (n_ctrl, n_state) feedback block using the clamp condition.
    tol = 1e-6
    clamped = jnp.repeat(((fwd - tol <= du_min) | (fwd + tol >= du_max))[:, None], n_state, axis=1)
    # --  Zero feedback for clamped controls. Original feedback for free controls.
    Bwd = jnp.where(clamped, jnp.zeros((n_ctrl, n_state)), -jnp.linalg.solve(Q_uu, Q_ux))

    # 4. Backward update for the value
    delta_V1 = fwd.T @ Q_u
    delta_V2 = jnp.linalg.multi_dot([fwd.T, Q_uu, fwd])
    V_x = Q_x + Q_ux.T @ fwd + Bwd.T @ Q_u + jnp.linalg.multi_dot([Bwd.T, Q_uu, fwd])
    V_xx = Q_xx + Bwd.T @ Q_ux + Q_ux.T @ Bwd + jnp.linalg.multi_dot([Bwd.T, Q_uu, Bwd])

    # 5. carry, accumulate
    return (fwd, V_x, V_xx), (fwd, Bwd, delta_V1, delta_V2)
