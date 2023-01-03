from itertools import combinations
import numpy as np

import jax
import jax.numpy as jnp


def multinomial_powers(n, k):
    """
    Returns all combinations of powers of the expansion (x_1+x_2+...+x_k)^n. The motivation for the algorithm
    is to use dots and bars:
    e.g.    For (x1+x2+x3)^3, count n=3 dots and k-1=2 bars.
            ..|.| = [x1^2, x2^1, x3^0]
    
    Notes:  1.  Add 1 variable (for a total of k+1) to include a constant term, (1+x+y+z)^n. This will yield
                all groups of powers less than or equal to n; then, ignore elem[0] to act only on x,y,z.

    :param n: the order of the multinomial_powers
    :param k: the number of variables {x_i}
    """
    for elem in combinations(np.arange(n + k - 1), k - 1):
        elem = np.array([-1] + list(elem) + [n + k - 1])
        yield elem[1:] - elem[:-1] - 1


def create_power_list(order, dimension):
    """
    Powers of the elements `x_1, x_2, ..., x_dimension` in all combinations less than or equal to `order`.
    See the docstring of `multinomial_powers`.

    The ordering guarantees the constant term (all zero exponents) is first, followed by powers in blocks
    of increasing totals. 
    
    For example, here is the output for `order=2`, `dimension=3`:

    ```
        [[0, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [2, 0, 0],
         [1, 1, 0],
         [1, 0, 1],
         [0, 2, 0],
         [0, 1, 1],
         [0, 0, 2]]
    ```

    :param order: the maximum of the control powers
    :param dimension: the number of control variables
    """
    return jnp.vstack([p[1:] for p in multinomial_powers(order, dimension + 1)][::-1])


def create_lifted_state_variable(x, u, powers):
    """
    Produce the lifted state variable `\Theta[u] \otimes x`, where `\Theta[u]` is the control to the 
    powers produced by `get_power_list`.

    Broadcast with a safe power function that explicitely handles `0^0` to prevent NaNs, see jax/issues/12409

    Notes:  1.  The `powers` param should be constructed before using this function within any
                JAX-differentiable code (because `multinomial_powers` uses non-differentiable code).
    """
    lifted_u = jnp.product(jnp.where(powers==0, 1., jnp.power(u, powers)), axis=1)
    return jnp.kron(lifted_u, x)


def fit_model(tx, tu, powers, rank):
    """
    Apply DMD.

    horiz + 1, n_state = tx.shape
    horiz, n_ctrl = tu.shape
    """
    X_2 = tx[1:].T
    XU_1 = jax.vmap(create_lifted_state_variable, in_axes=(0,0,None))(tx[:-1], tu, powers).T
    U, S, Vt = jnp.linalg.svd(XU_1)
    return X_2 @ Vt[:rank, :].conj().T @ jnp.diag(1 / S[:rank]) @  U[:, :rank].conj().T

def fit_model_split(tx2, tx1, tu, powers, rank):
    """
    Apply DMD with expicit past and future.

    horiz, n_state = tx2.shape
    horiz, n_state = tx1.shape
    horiz, n_ctrl = tu.shape
    """
    X_2 = tx2.T
    XU_1 = jax.vmap(create_lifted_state_variable, in_axes=(0,0,None))(tx1, tu, powers).T
    U, S, Vt = jnp.linalg.svd(XU_1)
    return X_2 @ Vt[:rank, :].conj().T @ jnp.diag(1 / S[:rank]) @  U[:, :rank].conj().T


def dmd_model(x, u, A_op, powers):
    """
    JAX-differentiable DMD model.
    """
    return A_op @ create_lifted_state_variable(x, u, powers)


def dmd_discrepancy_model(x, u, A_op, powers, x_guess, u_guess, prev_model):
    """
    JAX-differentiable DMD model.
    """
    return prev_model(x, u) + A_op @ create_lifted_state_variable(x - x_guess, u - u_guess, powers)
