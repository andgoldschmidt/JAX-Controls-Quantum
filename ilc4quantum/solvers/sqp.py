import functools
import jax
import jax.numpy as jnp
import jax.scipy.optimize as joptimize
from jax.scipy.linalg import block_diag
from jaxopt import BacktrackingLineSearch, BoxOSQP
from scipy.optimize import line_search as sp_line_search


sqp_static_args = ['max_iter', 'model_fn', 'linear_model_fn', 'cost_fn', 'approx_cost']


@functools.partial(jax.jit, static_argnames=sqp_static_args)
def sqp(
        x_init,
        tx_guess,
        tu_guess,
        model_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat,
        max_iter):
    """
    Sequential quadratic program.

    Notes:  1.  It is not necessary to use 'jax_enable_x32' for SQP.
            2.  It is important to have variables like max_iter last (else: "non-hashable static args...")
            
    :param x_init: Initial state
    :param tx_guess: Shape is (time, state)
    :param tu_guess: Shape is (time, control)
    :param model_fn: Model dynamics. Mapping z[t] -> z[t+1]
    :param linear_model_fn: Linearized model dynamics. Mapping z[t] -> z[t+1]
    :param cost_fn: Cost function (no terminal state cost). Mapping z[0:H-1] -> Reals
    :param approx_cost: Linearized cost function. Mapping z[0:H-1] -> Q[0:H-1], J[0:H-1]
    :param u_sat: Control saturation.
    :param du_sat: Slew rate.
    :param max_iter: Maximumum number of SQP steps.
    :return: The optimal trajectory, z[0:H]. A zero control is appended alongside the terminal state, z[H].
    :return: 
    """
    local_iteration = jax.tree_util.Partial(iteration_sqp,
                                            **{'x_init': x_init,
                                               'model_fn': model_fn,
                                               'linear_model_fn': linear_model_fn,
                                               'cost_fn': cost_fn,
                                               'approx_cost': approx_cost,
                                               'u_sat': u_sat,
                                               'du_sat': du_sat})
    # Scan
    (tx, tu), steps = jax.lax.scan(local_iteration, (tx_guess, tu_guess), None, length=max_iter)
    return jnp.concatenate([tx, jnp.vstack([tu, jnp.zeros(tu.shape[1])])], axis=1)


def iteration_sqp(
        carry,
        scan,
        x_init,
        model_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat):
    """
    Single QP iteration.

    Notes:  1.  Parameterized functions should be wrapped using jax.tree_util.Partial or declared as static args.
                (See jax/issues/1443)
    """
    tx_guess, tu_guess = carry

    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_guess.shape
    _, n_state = tx_guess.shape
    tz_guess = jnp.concatenate([tx_guess, jnp.vstack([tu_guess, jnp.zeros(n_ctrl)])], axis=1)

    # Linear expansion of model (and residual of guess)
    tr_feas = jax.vmap(model_fn)(tz_guess[:-1]) - tz_guess[1:, :n_state]
    tF_linear = jax.vmap(linear_model_fn)(tz_guess[:-1])

    # Quadratic expansion of cost about tz_guess
    tH_cost, tj_cost = approx_cost(tz_guess[:-1])

    # Solve QP
    tx_dx_opt, tu_dus_opt = quad_program(
        x_init=x_init,
        tx_g=tx_guess,
        tu_g=tu_guess,
        tF_lin=tF_linear,
        tr_feas=tr_feas,
        tH_cost=tH_cost,
        tj_cost=tj_cost,
        u_sat=u_sat,
        du_sat=du_sat
    )

    # Line search.
    tz_dz_opt = jnp.concatenate([tx_dx_opt, jnp.vstack([tu_dus_opt, jnp.zeros(n_ctrl)])], axis=1)
    # TODO: Cost function has no final state cost.
    step = traced_line_search(tz_guess[:-1], tz_dz_opt[:-1], cost_fn)
    tz_guess = tz_guess + step * tz_dz_opt
    return (tz_guess[:, :n_state], tz_guess[:-1, -n_ctrl:]), step


def full_search(tz_guess, tz_dz_opt, cost_fn):
    def cost_eval(alpha):
        return cost_fn(tz_guess + alpha * tz_dz_opt)

    opt = joptimize.minimize(cost_eval, jnp.array([0.5]), method='BFGS')
    return opt.x


def traced_line_search(tz_guess, tz_dz_opt, cost_fn, maxiter=10):
    n_step, n_dim = tz_guess.shape

    def wrap_cost_fn(vec_arg):
        return cost_fn(vec_arg.reshape(n_step, n_dim))

    ls = BacktrackingLineSearch(wrap_cost_fn, value_and_grad=False, maxiter=maxiter)
    step, state = ls.run(init_stepsize=1.,
                         params=tz_guess.flatten(),
                         descent_direction=tz_dz_opt.flatten())

    return step


def static_line_search(tz_guess, tz_dz_opt, cost_fn, maxiter=20):
    n_step, n_dim = tz_guess.shape

    def wrap_cost_fn(vec_arg):
        return cost_fn(vec_arg.reshape(n_step, n_dim))

    step, fc, gc, new_f, old_f, new_df = sp_line_search(wrap_cost_fn,
                                                        jax.grad(wrap_cost_fn),
                                                        tz_guess.flatten(),
                                                        tz_dz_opt.flatten(),
                                                        maxiter=maxiter)
    return step


def quad_program(
        x_init,
        tx_g,
        tu_g,
        tF_lin,
        tr_feas,
        tH_cost,
        tj_cost,
        u_sat,
        du_sat):
    """
        Solve an MPC iteration with a quadratic program.

        Cast MPC problem to a QP: x = [dx(0),du(0),dx(1),du(1),...,dx(H-1),du(H-1),dx(H)] where the parameter H is the
         horizon (H >= 1). All xs variables have length H + 1. Meanwhile, Hs_cost, Js_cost, and Fs_cost have length H.
         This means we must append (or pass an argument for) the appropriate x(H) action--we choose to append here.

        Notes for OSQP:
         * Careful with the signature of linalg.block_diag versus sparse.block_diag.
         * The arguments P and A are callable matvec objects (TODO: can these be made JAX sparse?)
    """
    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_g.shape
    _, n_state = tx_g.shape

    # - quadratic objective: (1/2)zHz
    tH_final = jnp.zeros_like(tH_cost[-1, :n_state, :n_state])
    P_qp = block_diag(*tH_cost, tH_final) / 2  # sparse.BCOO.fromdense?

    # - linear objective: Jz
    tJ_final = jnp.zeros_like(tj_cost[-1, :n_state])
    q_qp = jnp.hstack([tj_cost.flatten(), tJ_final])

    # - initial condition: dxs[0] == x_init - xs_g[0]
    I_eq = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_horiz * (n_state + n_ctrl)))])
    lo_eq_init = x_init - tx_g[0]
    up_eq_init = lo_eq_init

    # - linear dynamics: dxs[t+1] == F dzs[t] + rs[t]
    # dxs[t+1] = S dzs[t]
    id_x = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_ctrl))])
    S_eq = jnp.hstack([jnp.zeros((n_horiz * n_state, n_state + n_ctrl)),
                      block_diag(jnp.kron(jnp.eye(n_horiz - 1), id_x), jnp.eye(n_state))])
    # F dzs[t]
    F_eq = jnp.hstack([block_diag(*tF_lin), jnp.zeros((n_state * n_horiz, n_state))])
    # (S - F) dzs[t] := dxs[t+1] - F dzs[t]
    A_eq = S_eq - F_eq
    lo_eq = tr_feas.flatten()
    up_eq = lo_eq

    # - control inequality constraints
    broadcast = jnp.ones((n_horiz, n_ctrl))
    # -u < u_sat --> -u_g -(u - u_g) < u_sat
    # u < u_sat --> u_g + (u - u_g) < u_sat
    tu_min = -jnp.minimum(du_sat * broadcast, u_sat * broadcast + tu_g)
    tu_max = jnp.minimum(du_sat * broadcast, u_sat * broadcast - tu_g)

    # - state inequality constraints
    x_min = -jnp.inf
    x_max = jnp.inf

    # - combined inequality constraints
    Aineq = jnp.hstack([jnp.eye(n_horiz * (n_state + n_ctrl)),
                        jnp.zeros((n_horiz * (n_state + n_ctrl), n_state))])
    lo_ineq = jnp.concatenate((x_min * jnp.ones((n_horiz, n_state)), tu_min), axis=1).flatten()
    up_ineq = jnp.concatenate((x_max * jnp.ones((n_horiz, n_state)), tu_max), axis=1).flatten()

    # - OSQP constraints
    A_qp = jnp.vstack([I_eq, A_eq, Aineq])  # sparse.BCOO.fromdense?
    lo_qp = jnp.hstack([lo_eq_init, lo_eq, lo_ineq])
    up_qp = jnp.hstack([up_eq_init, up_eq, up_ineq])

    # - OSQP Solve
    # If required the algorithm can be sped up by setting check_primal_dual_infeasability to False,
    # and by setting eq_qp_preconditioner to "jacobi" (when possible).
    prob = BoxOSQP()
    params, state = prob.run(params_obj=(P_qp, q_qp), params_eq=A_qp, params_ineq=(lo_qp, up_qp))

    # params.primal[0] = [dx(0),du(0),dx(1),du(1),...,dx(H-1),du(H-1),dx(H)
    res = jnp.hstack([params.primal[0], jnp.zeros(n_ctrl)]).reshape(n_horiz + 1, n_state + n_ctrl)
    return res[:, :n_state], res[:-1, -n_ctrl:]
