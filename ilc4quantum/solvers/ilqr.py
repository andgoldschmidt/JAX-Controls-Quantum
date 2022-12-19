import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP, BacktrackingLineSearch

from .solver import register


@register("iLQR")
def iteration_lqr(
        carry,
        iteration,
        x_init,
        u_init,
        model_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat):
    """
    Iterative LQR with box constraints (1st order in dynamics, 2nd order in cost).

    Read more:
    * 10.1109/ICRA.2014.6907001
    *

    Notes:  1.  It is recommended to use 'jax_enable_x64' for iLQR.
            2.  Slew bounds must account for the rate from the control guess.
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
    # -- Run scan backward
    # TODO: Search mu?
    _, feeds = jax.lax.scan(jax.tree_util.Partial(scan_backward, mu=1e-3, u_sat=u_sat, du_sat=du_sat, prob=prob),
                            (fwd, V_x, V_xx),
                            (tu_guess, tu_guess_slew, tj_cost, tH_cost, tF_linear),
                            reverse=True)
    feedforwards, Feedbacks, delta_V1s, delta_V2s = feeds

    # 3. Forward pass
    forward_pass_step = jax.tree_util.Partial(
        forward_pass,
        tz_guess=tz_guess,
        tu_feedforward=feedforwards,
        tux_Feedback=Feedbacks,
        x_init=x_init,
        u_init=u_init,
        model_fn=model_fn,
        u_sat=u_sat,
        du_sat=du_sat)

    # -- Line search parameters
    # cost_guess = cost_fn(tz_guess[1:])
    # # wolfe_c1 = 0.0001
    # dcost_lin = jnp.sum(delta_V1s, axis=0)
    # # wolfe_c2 = 0.9
    # # dcost_quad = jnp.sum(delta_V2s, axis=0)
    # step = jax.lax.while_loop(
    #     jax.tree_util.Partial(backtracking_line_search,
    #                           forward_pass_step=forward_pass_step,
    #                           cost_fn=cost_fn,
    #                           cost_guess=cost_guess,
    #                           dcost_lin=dcost_lin),
    #     body_fn,
    #     1.
    # )
    # TODO: I get better results taking a full step than I do following the line search. Why?
    step = 1.

    # -- Take step
    (x_end, _), tz_opt = forward_pass_step(step)
    tx_next = jnp.vstack([tz_opt[:, :n_state], x_end])
    tu_next = tz_opt[:, -n_ctrl:]
    return (tx_next, tu_next), step


def backtracking_line_search(
        step,
        forward_pass_step,
        cost_fn,
        cost_guess,
        dcost_lin,
        min_step=1e-5,
        wolfe_c1=1e-4):
    """
    The backtracking line search starts begins with an appropriate initial step length (like 1 for Newton's method).
    The algorithm greedily checks for satisfaction of the Armijo sufficient decrease condition.

    :param step: The step size, u + step * f + F dx
    :param forward_pass_step: Partial function mapping: step --> (x_end, u_end), tz_opt
    :param cost_fn:
    :param cost_guess:
    :param dcost_lin:
    :param min_step: Smallest allowed step size
    :param wolfe_c1: Armijo (sufficient decrease) condition
    :return: True if step satisifies the backtracking line search. Else false.
    """
    _, zs = forward_pass_step(step)
    # -- Backtracking line search: Exit when true.
    armijo_cond = (step > min_step) & (cost_fn(zs) > cost_guess + wolfe_c1 * step * dcost_lin)
    return armijo_cond


# TODO: What discount rate?
def body_fn(s, discount=0.8):
    return s * discount


def forward_pass(
        step,
        tz_guess,
        tu_feedforward,
        tux_Feedback,
        x_init,
        u_init,
        model_fn,
        u_sat,
        du_sat):
    """
    Run `scan_forward` for a fixed step size. We use this to construct `forward_pass_step` based on the results of the
    backward iteration.
    """
    return jax.lax.scan(
        jax.tree_util.Partial(scan_forward, step=step, u_sat=u_sat, du_sat=du_sat, model_fn=model_fn),
        (x_init, u_init),
        (tz_guess[:-1], tu_feedforward, tux_Feedback)
    )


def scan_forward(carry, scan, step, u_sat, du_sat, model_fn):
    x, u_prev = carry
    z_guess, feedforward, Feedback = scan
    n_ctrl, n_state = Feedback.shape

    # Clip on slew and saturation
    #  -- This is naive clamping, but in the backward pass 1. feedforward is constrained, and 2. Feedback is zeroed
    #     along any constrained dimensions.
    u = z_guess[-n_ctrl:] + step * feedforward + Feedback @ (x - z_guess[:n_state])
    u = jnp.clip(u, a_min=jnp.maximum(u_prev - du_sat, -u_sat), a_max=jnp.minimum(du_sat + u_prev, u_sat))
    return (model_fn(x, u), u), jnp.hstack((x, u))


def scan_backward(carry, scan, mu, u_sat, du_sat, prob):
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
