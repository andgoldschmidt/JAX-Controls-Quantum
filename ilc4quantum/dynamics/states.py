import functools
import jax
import jax.numpy as jnp


def commutator(a, b):
    return a @ b - b @ a


def dag(a):
    return a.conj().T


def structure_constant(i, j, k, basis):
    return jnp.trace(dag(commutator(basis[i], basis[j])) @ basis[k])
    # jnp.where(i == j, 0.0, jnp.trace(dag(commutator(basis[i], basis[j])) @ basis[k]))


@jax.jit
def _vectorize_me(H, basis):
    dim_b = len(basis)

    # Precompute the structure constants in the measurement basis
    i3, j3, k3 = jnp.mgrid[0:dim_b, 0:dim_b, 0:dim_b]
    structure_table = jnp.vectorize(structure_constant, excluded=(3,))(i3, j3, k3, basis)
    # structure_table = jax.vmap(jax.vmap(jax.vmap(
    #     structure_constant,
    #     in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None)
    # )(i3, j3, k3, basis)

    # Precompute the Hamiltonian projection in the measurement basis
    H_list = jax.vmap(lambda sigma_i: jnp.trace(dag(H) @ sigma_i))(basis)

    # Project the Liouville equation operator (adjoint action)
    # A_op[j, k] = -1j * sum(H_list[i] * structure_table[i, k, j], i)
    A_op = jnp.sum(-1j * H_list[:, None, None] * jnp.swapaxes(structure_table, -2, -1), axis=0)

    # Real-valued result
    return jnp.block([[A_op.real, -A_op.imag], [A_op.imag, A_op.real]])


def vectorize_me(H, basis, is_complex=False):
    """
    Expand the master equation $\dot{\rho} = -i[H, \rho]$ using the given basis. Let $r_j := \Tr{b_j \rho}$) be the
    coordinates of the state in the given basis, so $\vec{r} = [r_1, r_2, \dots, r_{dim_b}]$. Then this function
    returns the linear operator A such that $\dot{\vec{r}} = A \vec{r}$.

    Parameters:
        H: Hamiltonian
        basis: a list of operators (of size matching H) to project the master equation.
        is_complex: whether $\vec{r}$ is real-valued or complex. If complex,
                 $\vec{r} \mapsto \text{stack}[\text{Real}[\vec{r}], \text{Imag}[\vec{r}]]$.

    For example, if measure_list are the Pauli matrices, then H is mapped to an equivalent linear operator that
    describes the dynamics of the Bloch vector. The Bloch vector is real-valued, so we can return the real part
    of this linear operator.
    """
    dim_b = len(basis)
    A_blocks = _vectorize_me(H, basis)
    return A_blocks if is_complex else A_blocks[:dim_b, :dim_b]