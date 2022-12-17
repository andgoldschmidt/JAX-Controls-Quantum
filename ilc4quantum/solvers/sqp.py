import jax
from jax.experimental import sparse
import jax.numpy as jnp
import jax.scipy.optimize as joptimize
from jax.scipy.linalg import block_diag
from jaxopt import BacktrackingLineSearch, BoxOSQP
from scipy.optimize import line_search as sp_line_search

from .solver import register


@register("SQP")
def iteration_sqp(
        carry,
        scan,
        x_init,
        u_init,
        model_fn,
        linear_model_fn,
        cost_fn,
        approx_cost,
        u_sat,
        du_sat):
    """
    Single quadratic program iteration.

    Notes:  1.  Parameterized functions should be wrapped using jax.tree_util.Partial or declared as static args.
                (See jax/issues/1443)
            2.  It is not necessary to use 'jax_enable_x32' for SQP.
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
        u_init=u_init,
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
        u_init,
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
         * The arguments P and A can be parameters to callable matvec functions (allowing sparse). BCOO matrices must
           specify a concrete number for nse (number of specified elements).
    """
    # Pay attention to shapes!
    n_horiz, n_ctrl = tu_g.shape
    _, n_state = tx_g.shape

    # - quadratic objective: (1/2)zHz
    tH_final = jnp.zeros_like(tH_cost[-1, :n_state, :n_state])
    P_qp = block_diag(*tH_cost, tH_final) / 2
    nse_P_qp = n_horiz * (n_state + n_ctrl) ** 2 + n_state ** 2

    # - linear objective: Jz
    tJ_final = jnp.zeros_like(tj_cost[-1, :n_state])
    q_qp = jnp.hstack([tj_cost.flatten(), tJ_final])

    # - initial condition: dx[0] == x_init - x_g[0]
    # TODO: Allow for initial control specification?
    I_x_init = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_horiz * (n_state + n_ctrl)))])
    nse_I_x_eq = n_state
    lo_x_init = x_init - tx_g[0]
    up_x_init = lo_x_init

    # - initial slew:
    # u[0] - u_init <= du_sat --> u[0] - u_g[0] + u_g[0] - u_init <= du_sat --> du[0] = du_sat - slew_init
    # -(u[0] - u_init) <= du_sat --> du[0] >= -(du_sat + slew_init)
    slew_init = tu_g[0] - u_init
    I_u_init = jnp.hstack([
        jnp.zeros((n_ctrl, n_state)), jnp.eye(n_ctrl),
        jnp.zeros((n_ctrl, (n_horiz - 1) * (n_state + n_ctrl))), jnp.zeros((n_ctrl, n_state))
    ])
    nse_I_u_ineq = n_ctrl
    lo_u_init = -(du_sat + slew_init)
    up_u_init = du_sat - slew_init

    # - linear dynamics: dx[t+1] == F dz[t] + r[t]
    # dx[t+1] = S dz[t]
    id_x = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_ctrl))])
    S_eq = jnp.hstack([
        jnp.zeros((n_horiz * n_state, n_state + n_ctrl)),
        block_diag(jnp.kron(jnp.eye(n_horiz - 1), id_x), jnp.eye(n_state))
    ])
    # F dzs[t]
    F_eq = jnp.hstack([block_diag(*tF_lin), jnp.zeros((n_state * n_horiz, n_state))])
    # (S - F) dz[t] := dx[t+1] - F dz[t]
    A_eq = S_eq - F_eq
    nse_A_eq = n_horiz * (n_state + n_ctrl) ** 2 + n_state ** 2
    lo_eq = tr_feas.flatten()
    up_eq = lo_eq

    # - control inequality constraints
    # -u < u_sat --> -u_g -(u - u_g) < u_sat
    # u < u_sat --> u_g + (u - u_g) < u_sat
    tu_du_min = -(u_sat * jnp.ones((n_horiz, n_ctrl)) + tu_g)
    tu_du_max = u_sat * jnp.ones((n_horiz, n_ctrl)) - tu_g

    # - state inequality constraints
    x_min = -jnp.inf
    x_max = jnp.inf

    # - combined saturation inequality constraints
    A_ineq_sat = jnp.hstack([
        jnp.eye(n_horiz * (n_state + n_ctrl)), jnp.zeros((n_horiz * (n_state + n_ctrl), n_state))
    ])
    # Pair up each (dx_sat(t), du_sat(t)), then flatten.
    lo_ineq_sat = jnp.concatenate((x_min * jnp.ones((n_horiz, n_state)), tu_du_min), axis=1).flatten()
    up_ineq_sat = jnp.concatenate((x_max * jnp.ones((n_horiz, n_state)), tu_du_max), axis=1).flatten()
    nse_A_ineq_sat = n_horiz * (n_state + n_ctrl)

    # - control slew inequality constraints
    slew_block_minus = jnp.kron(
        jnp.eye(n_horiz), jnp.hstack([jnp.zeros((n_ctrl, n_state)), -jnp.eye(n_ctrl)])
    )[:-n_ctrl]
    slew_block_plus = jnp.kron(
        jnp.eye(n_horiz), jnp.hstack([jnp.zeros((n_ctrl, n_state)), jnp.eye(n_ctrl)])
    )[n_ctrl:]
    A_ineq_slew = jnp.concatenate([slew_block_minus + slew_block_plus,
                                   jnp.zeros((n_ctrl * (n_horiz - 1), n_state))], axis=1)
    tu_g_slew = tu_g[1:] - tu_g[:-1]
    lo_ineq_slew = -(du_sat * jnp.ones((n_horiz - 1, n_ctrl)) + tu_g_slew).flatten()
    up_ineq_slew = (du_sat * jnp.ones((n_horiz - 1, n_ctrl)) - tu_g_slew).flatten()
    nse_A_ineq_slew = 2 * n_ctrl * (n_horiz - 1)

    # - OSQP constraints
    A_qp = jnp.vstack([I_x_init, I_u_init, A_eq, A_ineq_sat, A_ineq_slew])
    lo_qp = jnp.hstack([lo_x_init, lo_u_init, lo_eq, lo_ineq_sat, lo_ineq_slew])
    up_qp = jnp.hstack([up_x_init, up_u_init, up_eq, up_ineq_sat, up_ineq_slew])

    # - OSQP Solve
    # If required the algorithm can be sped up by setting check_primal_dual_infeasability to False,
    # and by setting eq_qp_preconditioner to "jacobi" (when possible).
    prob = BoxOSQP(matvec_Q=sparse_matvec, matvec_A=sparse_matvec)
    sp_P_qp = sparse.BCOO.fromdense(P_qp, nse=nse_P_qp)
    sp_A_qp = sparse.BCOO.fromdense(
        A_qp,
        nse=nse_I_x_eq + nse_I_u_ineq + nse_A_eq + nse_A_ineq_sat + nse_A_ineq_slew
    )
    params, state = prob.run(params_obj=(sp_P_qp, q_qp), params_eq=sp_A_qp, params_ineq=(lo_qp, up_qp))

    # params.primal[0] = [dx(0),du(0),dx(1),du(1),...,dx(H-1),du(H-1),dx(H)
    res = jnp.hstack([params.primal[0], jnp.zeros(n_ctrl)]).reshape(n_horiz + 1, n_state + n_ctrl)
    return res[:, :n_state], res[:-1, -n_ctrl:]


# def quad_program(
#         x_init,
#         tx_g,
#         tu_g,
#         tF_lin,
#         tr_feas,
#         tH_cost,
#         tj_cost,
#         u_sat,
#         du_sat):
#     """
#         Solve an MPC iteration with a quadratic program.
#
#         Cast MPC problem to a QP: x = [dx(0),du(0),dx(1),du(1),...,dx(H-1),du(H-1),dx(H)] where the parameter H is the
#          horizon (H >= 1). All xs variables have length H + 1. Meanwhile, Hs_cost, Js_cost, and Fs_cost have length H.
#          This means we must append (or pass an argument for) the appropriate x(H) action--we choose to append here.
#
#         Notes for OSQP:
#          * Careful with the signature of linalg.block_diag versus sparse.block_diag.
#          * The arguments P and A can be parameters to callable matvec functions (allowing sparse).
#     """
#     # Pay attention to shapes!
#     n_horiz, n_ctrl = tu_g.shape
#     _, n_state = tx_g.shape
#
#     # - quadratic objective: (1/2)zHz
#     tH_final = jnp.zeros_like(tH_cost[-1, :n_state, :n_state])
#     P_qp = block_diag(*tH_cost, tH_final) / 2  # sparse.BCOO.fromdense?
#     nse_P_qp = n_horiz * (n_state + n_ctrl) ** 2 + n_state ** 2
#
#     # - linear objective: Jz
#     tJ_final = jnp.zeros_like(tj_cost[-1, :n_state])
#     q_qp = jnp.hstack([tj_cost.flatten(), tJ_final])
#
#     # - initial condition: dxs[0] == x_init - xs_g[0]
#     I_eq = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_horiz * (n_state + n_ctrl)))])
#     nse_I_eq = n_state
#     lo_eq_init = x_init - tx_g[0]
#     up_eq_init = lo_eq_init
#
#     # - linear dynamics: dxs[t+1] == F dzs[t] + rs[t]
#     # dxs[t+1] = S dzs[t]
#     id_x = jnp.hstack([jnp.eye(n_state), jnp.zeros((n_state, n_ctrl))])
#     S_eq = jnp.hstack([jnp.zeros((n_horiz * n_state, n_state + n_ctrl)),
#                        block_diag(jnp.kron(jnp.eye(n_horiz - 1), id_x), jnp.eye(n_state))])
#     # F dzs[t]
#     F_eq = jnp.hstack([block_diag(*tF_lin), jnp.zeros((n_state * n_horiz, n_state))])
#     # (S - F) dzs[t] := dxs[t+1] - F dzs[t]
#     A_eq = S_eq - F_eq
#     nse_A_eq = n_horiz * (n_state + n_ctrl) ** 2 + n_state ** 2
#     lo_eq = tr_feas.flatten()
#     up_eq = lo_eq
#
#     # - control inequality constraints
#     broadcast = jnp.ones((n_horiz, n_ctrl))
#     # -u < u_sat --> -u_g -(u - u_g) < u_sat
#     # u < u_sat --> u_g + (u - u_g) < u_sat
#     tu_min = -jnp.minimum(du_sat * broadcast, u_sat * broadcast + tu_g)
#     tu_max = jnp.minimum(du_sat * broadcast, u_sat * broadcast - tu_g)
#
#     # - state inequality constraints
#     x_min = -jnp.inf
#     x_max = jnp.inf
#
#     # - combined inequality constraints
#     Aineq = jnp.hstack([jnp.eye(n_horiz * (n_state + n_ctrl)),
#                         jnp.zeros((n_horiz * (n_state + n_ctrl), n_state))])
#     lo_ineq = jnp.concatenate((x_min * jnp.ones((n_horiz, n_state)), tu_min), axis=1).flatten()
#     up_ineq = jnp.concatenate((x_max * jnp.ones((n_horiz, n_state)), tu_max), axis=1).flatten()
#     nse_A_ineq = n_horiz * (n_state + n_ctrl)
#
#     # - OSQP constraints
#     A_qp = jnp.vstack([I_eq, A_eq, Aineq])  # sparse.BCOO.fromdense?
#     lo_qp = jnp.hstack([lo_eq_init, lo_eq, lo_ineq])
#     up_qp = jnp.hstack([up_eq_init, up_eq, up_ineq])
#
#     # - OSQP Solve
#     # If required the algorithm can be sped up by setting check_primal_dual_infeasability to False,
#     # and by setting eq_qp_preconditioner to "jacobi" (when possible).
#     prob = BoxOSQP(matvec_Q=sparse_matvec, matvec_A=sparse_matvec)
#     sp_P_qp = sparse.BCOO.fromdense(P_qp, nse=nse_P_qp)
#     sp_A_qp = sparse.BCOO.fromdense(A_qp, nse=nse_I_eq + nse_A_eq + nse_A_ineq)
#     params, state = prob.run(params_obj=(sp_P_qp, q_qp), params_eq=sp_A_qp, params_ineq=(lo_qp, up_qp))
#
#     # params.primal[0] = [dx(0),du(0),dx(1),du(1),...,dx(H-1),du(H-1),dx(H)
#     res = jnp.hstack([params.primal[0], jnp.zeros(n_ctrl)]).reshape(n_horiz + 1, n_state + n_ctrl)
#     return res[:, :n_state], res[:-1, -n_ctrl:]


@sparse.sparsify
def sparse_matvec(M, v):
    return M @ v
