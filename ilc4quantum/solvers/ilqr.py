import functools
import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP


ilqr_static_args = ['max_iter', 'model_fn', 'linearize_model', 'cost_fn', 'quadraticize_cost']


@functools.partial(jax.jit, static_argnames=ilqr_static_args)
def ilqr(
        x_init,
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
    Iterative LQR with box constraints (1st order in dynamics, 2nd order in cost).

    Notes:  1.  It is recommended to use 'jax_enable_x64' for iLQR.
            2.  Slew bounds must account for the rate from the control guess.

    :param x_init: Initial state
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
    local_iteration = jax.tree_util.Partial(iteration_lqr,
                                            **{'x_init': x_init,
                                               'model_fn': model_fn,
                                               'linear_model_fn': linearize_model,
                                               'cost_fn': cost_fn,
                                               'approx_cost': quadraticize_cost,
                                               'u_sat': u_sat,
                                               'du_sat': du_sat})
    (tx, tu), steps = jax.lax.scan(local_iteration, (tx_guess, tu_guess), None, length=max_iter)
    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1)


def iteration_lqr(
        carry,
        iteration,
        x_init,
        model_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat):
    """
    Single LQR iteration.
    """
    tx_guess, tu_guess = carry

    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess.shape
    _, n_state = tx_guess.shape
    tz_guess = jnp.concatenate([tx_guess, jnp.vstack([tu_guess, jnp.zeros(n_ctrl)])], axis=1)

    # Accumulated slew
    # TODO: Initial slew?
    slew_init = jnp.zeros(n_ctrl)
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
    # -- Line search parameters
    min_step = 1e-3
    cost_guess = cost_fn(tz_guess[1:])
    wolfe_c1 = 0.0001
    dcost_lin = jnp.sum(delta_V1s, axis=0)
    # wolfe_c2 = 0.9
    # dcost_quad = jnp.sum(delta_V2s, axis=0)

    local_cond_fn = jax.tree_util.Partial(
        cond_fn,
        x_init=x_init,
        scan=(tz_guess[:-1], tu_guess_slew, feedforwards, Feedbacks),
        u_sat=u_sat,
        du_sat=du_sat,
        model_fn=model_fn,
        cost_fn=cost_fn,
        cost_guess=cost_guess,
        dcost_lin=dcost_lin,
        wolfe_c1=wolfe_c1,
        min_step=min_step
    )

    # TODO: How to improve the n_eval here?
    step = jax.lax.while_loop(local_cond_fn, body_fn, 1.)
    step = jnp.where(step <= min_step, 0., step)
    x_end, zs_opt = jax.lax.scan(
        jax.tree_util.Partial(scan_forward, step=step, u_sat=u_sat, du_sat=du_sat, model_fn=model_fn),
        x_init,
        (tz_guess[:-1], tu_guess_slew, feedforwards, Feedbacks)
    )
    xs_guess_next = jnp.vstack([zs_opt[:, :n_state], x_end])
    us_guess_next = zs_opt[:, -n_ctrl:]
    return (xs_guess_next, us_guess_next), step


def cond_fn(s, x_init, scan, u_sat, du_sat, model_fn, cost_fn, cost_guess, dcost_lin, wolfe_c1, min_step):
    # scan = (zs_guess[:-1], feedforwards, Feedbacks)
    _, zs = jax.lax.scan(
        jax.tree_util.Partial(scan_forward, step=s, u_sat=u_sat, du_sat=du_sat, model_fn=model_fn), x_init, scan
    )
    armijo_cond = (s > min_step) & (cost_fn(zs) > cost_guess + wolfe_c1 * s * dcost_lin)
    return armijo_cond


# TODO: What discount rate?
def body_fn(s, discount=0.5):
    return s * discount


def scan_forward(x, scan, step, u_sat, du_sat, model_fn):
    z_guess, u_guess_slew, feedforward, Feedback = scan
    n_ctrl, n_state = Feedback.shape

    # Clip
    # TODO: Is |fwd| <= du_sat? Would Feedback ever exceed clip?
    du = jnp.clip(
        step * feedforward + Feedback @ (x - z_guess[:n_state]),
        a_min=-jnp.minimum(du_sat + u_guess_slew, u_sat + z_guess[-n_ctrl:]),
        a_max=jnp.minimum(du_sat - u_guess_slew, u_sat - z_guess[-n_ctrl:])
    )
    z = jnp.hstack((x, z_guess[-n_ctrl:] + du))
    return model_fn(z), z


def scan_backward(carry, scan, mu, u_sat, du_sat, prob):
    next_fwd, next_V_x, next_V_xx = carry
    u_guess, u_guess_slew, J_cost, H_cost, F_linear = scan
    # Pay attention to shapes!
    n_ctrl = len(u_guess)
    n_state = len(next_V_x)

    # 1. Approximate the Bellman value function
    # TODO: Make this DDP for faster convergence
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
    # -- Solve the box QP (warm start with prev. step) (fwd = -Q_uu_inv @ Q_u)
    fwd, state = prob.run(next_fwd, params_obj=(Q_uu, Q_u), params_ineq=(du_min, du_max))

    # 3. Compute Feedback
    # -- Feedback for free controls. Zero for clamped.
    clamped = jnp.repeat(((fwd <= du_min) | (fwd >= du_max))[:, None], n_state, axis=1)
    Bwd = jnp.where(clamped, jnp.zeros((n_ctrl, n_state)), -jnp.linalg.pinv(Q_uu) @ Q_ux)

    # 4. Backward update for the value
    delta_V1 = fwd.T @ Q_u
    delta_V2 = jnp.linalg.multi_dot([fwd.T, Q_uu, fwd])
    V_x = Q_x + Q_ux.T @ fwd + Bwd.T @ Q_u + jnp.linalg.multi_dot([Bwd.T, Q_uu, fwd])
    V_xx = Q_xx + Bwd.T @ Q_ux + Q_ux.T @ Bwd + jnp.linalg.multi_dot([Bwd.T, Q_uu, Bwd])

    # 5. carry, accumulate
    return (fwd, V_x, V_xx), (fwd, Bwd, delta_V1, delta_V2)
