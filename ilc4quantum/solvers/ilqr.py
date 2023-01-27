import jax
import jax.numpy as jnp

from .solver import register


@register("iLQR")
def iteration_lqr(
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
        Run iterative LQR (1st order in dynamics, 2nd order in cost).

        Notes:  1.  NOTE/TODO: We can put the slew and saturation into the cost function by changing the definition of
                    the state, and using cost matrices to make the constraint.

        Read more:  *  10.1109/IROS.2012.6386025
    """
    tx_guess, tu_guess = carry

    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess.shape
    _, n_state = tx_guess.shape
    tz_guess = jnp.concatenate([tx_guess, jnp.vstack([tu_guess, jnp.zeros(n_ctrl)])], axis=1)

    # 1. Compute derivatives
    # -- Linear expansion of model
    tF_linear = jax.vmap(linear_model_fn)(tz_guess[:-1])
    # -- Quadratic expansion of cost about tz_guess
    tH_cost, tj_cost = approx_cost(tz_guess[:-1])

    # 2. Backward pass
    V_x = tj_cost[-1, :n_state]
    V_xx = tH_cost[-1, :n_state, :n_state]
    _, feeds = jax.lax.scan(jax.tree_util.Partial(scan_backward, mu=1e-9),
                            (V_x, V_xx),
                            (tu_guess, tj_cost, tH_cost, tF_linear),
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
    cost_guess = cost_fn(tz_guess[:-1])
    dcost_lin = jnp.sum(delta_V1s, axis=0)
    step = jax.lax.while_loop(
        jax.tree_util.Partial(backtracking_line_search,
                              forward_pass_step=model_step,
                              cost_fn=cost_fn,
                              cost_guess=cost_guess,
                              dcost_lin=dcost_lin),
        discount_fn,
        1.
    )
    # step = 1.

    # -- Take step
    (x_end, _), tz_opt = rollout_step(step)
    tx_next = jnp.vstack([tz_opt[:, :n_state], x_end])
    tu_next = tz_opt[:, -n_ctrl:]
    return (tx_next, tu_next), (step, jnp.max(jnp.abs(tu_feedforward), axis=0))


def scan_backward(carry, scan, mu):
    # TODO: Search mu?
    next_V_x, next_V_xx = carry
    u_guess, J_cost, H_cost, F_linear = scan
    # Pay attention to shapes!
    n_ctrl = len(u_guess)
    n_state = len(next_V_x)

    # 1. Approximate the Bellman value function
    F_x = F_linear[:, :n_state]
    F_u = F_linear[:, -n_ctrl:]
    Q_u = J_cost[-n_ctrl:] + F_u.T @ next_V_x
    Q_x = J_cost[:n_state] + F_x.T @ next_V_x
    Q_xx = H_cost[:n_state, :n_state] + jnp.linalg.multi_dot([F_x.T, next_V_xx, F_x])
    # -- Regularize with a quadratic state-cost around the previous sequence.
    Q_uu = H_cost[-n_ctrl:, -n_ctrl:] + jnp.linalg.multi_dot([F_u.T, next_V_xx + mu * jnp.identity(n_state), F_u])
    Q_ux = H_cost[-n_ctrl:, :n_state] + jnp.linalg.multi_dot([F_u.T, next_V_xx + mu * jnp.identity(n_state), F_x])

    # 2. Compute feedback
    fwd = -jnp.linalg.solve(Q_uu, Q_u)
    Bwd = -jnp.linalg.solve(Q_uu, Q_ux)

    # 3. Backward update for the value
    delta_V1 = fwd.T @ Q_u
    delta_V2 = jnp.linalg.multi_dot([fwd.T, Q_uu, fwd])
    V_x = Q_x + Q_ux.T @ fwd + Bwd.T @ Q_u + jnp.linalg.multi_dot([Bwd.T, Q_uu, fwd])
    V_xx = Q_xx + Bwd.T @ Q_ux + Q_ux.T @ Bwd + jnp.linalg.multi_dot([Bwd.T, Q_uu, Bwd])

    # 4. carry, accumulate
    return (V_x, V_xx), (fwd, Bwd, delta_V1, delta_V2)


def scan_forward(carry, scan, step, u_sat, du_sat, model_fn):
    x, u_prev = carry
    z_guess, feedforward, Feedback = scan
    n_ctrl, n_state = Feedback.shape

    # Clip on slew and saturation
    #  -- This is naive clamping.
    u = z_guess[-n_ctrl:] + step * feedforward + Feedback @ (x - z_guess[:n_state])
    u = jnp.clip(u, a_min=jnp.maximum(u_prev - du_sat, -u_sat), a_max=jnp.minimum(du_sat + u_prev, u_sat))
    return (model_fn(x, u), u), jnp.hstack((x, u))


def forward_pass(
        step,
        scan_forward_step,
        carry_init,
        scan):
    """
    Run `scan_forward` for a fixed step size. We use this to construct `forward_pass_step` based on the results of the
    backward iteration---this is an argument in the `backtracking_line_search` function.
    """
    return jax.lax.scan(jax.tree_util.Partial(scan_forward_step, step=step), carry_init, scan)


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
    _, tz_step = forward_pass_step(step)
    # -- Backtracking line search: Exit when true.
    armijo_cond = (step > min_step) & (cost_fn(tz_step) > cost_guess + wolfe_c1 * step * dcost_lin)
    return armijo_cond


def discount_fn(s, discount=0.8):
    # TODO: What discount rate?
    return s * discount
