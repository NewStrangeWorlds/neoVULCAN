"""Block-tridiagonal LU solver for the Rosenbrock LHS, in JAX.

The LHS ``W = c0*I - dfdy`` of every Rosenbrock stage is block-tridiagonal
over the vertical grid: dense ``ni x ni`` diagonal blocks (chemistry couples
species within a layer) and *diagonal* off-diagonal blocks (transport couples
neighbouring layers species by species).  A block Thomas sweep with a pivoted
dense LU per layer solves it in roughly half the time of the general banded
LAPACK factorisation (``dgbtrf`` at bandwidth ``2*ni-1``), and the
factorisation is reused for every stage of a step exactly as before.

Storage (see :func:`jacobian_jax._lhs_jac_blocks_kernel`):

    diag  : (nz, ni, ni)   D_iz
    sup_d : (nz-1, ni)     diagonal of U_iz     (row block iz,   column block iz+1)
    sub_d : (nz-1, ni)     diagonal of L_{iz+1} (row block iz+1, column block iz)

so layer ``j`` of ``W x = b`` reads ``L_j x_{j-1} + D_j x_j + U_j x_{j+1} = b_j``.

Forward elimination (Schur complements, pivoted LU per layer):

    A_0 = D_0
    A_j = D_j - L_j A_{j-1}^{-1} U_{j-1}          j = 1 .. nz-1

Because ``L_j`` and ``U_{j-1}`` are diagonal the correction is
``l_j[:, None] * (A_{j-1}^{-1} diag(u_{j-1}))`` -- one multi-RHS triangular
solve, no matrix product.  Pivoting inside each ``A_j`` is what keeps the
sweep stable at the conditioning the chemistry produces (``c0`` tiny against
loss rates of 1e17 s^-1); do not replace it with an explicit inverse carried
along the sweep.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve

jax.config.update("jax_enable_x64", True)


@jax.jit
def apply_fixed_rows(diag, sup_d, sub_d, mask, c0):
    """Replace the rows of frozen unknowns by ``c0`` times unit rows.

    ``mask`` is a boolean ``(nz, ni)`` array marking (layer, species)
    unknowns to freeze (fixed condensable species, electrons).  With the
    matching RHS entries zeroed the solve returns a zero increment for them
    while all other unknowns keep their full Jacobian coupling -- the
    block-storage twin of :func:`ode_solver.zero_rows_banded`.
    """
    ni = diag.shape[1]
    di = jnp.arange(ni)
    diag = jnp.where(mask[:, :, None], 0.0, diag)
    diag = diag.at[:, di, di].set(jnp.where(mask, c0, diag[:, di, di]))
    sup_d = jnp.where(mask[:-1], 0.0, sup_d)     # U_iz[s, s]     for frozen (iz, s), iz <= nz-2
    sub_d = jnp.where(mask[1:], 0.0, sub_d)      # L_iz[s, s]     for frozen (iz, s), iz >= 1
    return diag, sup_d, sub_d


@jax.jit
def factor_block_tridiag(diag, sup_d, sub_d):
    """Forward elimination.  Returns ``(lu, piv, sup_d, sub_d)`` where
    ``lu``/``piv`` (``(nz, ni, ni)`` / ``(nz, ni)``) are the pivoted LU
    factors of the Schur complements ``A_j``.  The off-diagonals are passed
    through so the result is a self-contained operator for
    :func:`solve_block_tridiag`."""
    lu0, piv0 = lu_factor(diag[0])

    def step(carry, inp):
        lu_p, piv_p = carry
        D_j, u_jm1, l_j = inp
        X = lu_solve((lu_p, piv_p), jnp.diag(u_jm1))      # A_{j-1}^{-1} U_{j-1}
        A_j = D_j - l_j[:, None] * X                       # L_j is diagonal
        lu_j, piv_j = lu_factor(A_j)
        return (lu_j, piv_j), (lu_j, piv_j)

    _, (lu_tail, piv_tail) = lax.scan(step, (lu0, piv0), (diag[1:], sup_d, sub_d))
    lu = jnp.concatenate([lu0[None], lu_tail], axis=0)
    piv = jnp.concatenate([piv0[None], piv_tail], axis=0)
    return lu, piv, sup_d, sub_d


@jax.jit
def solve_block_tridiag(lu, piv, sup_d, sub_d, rhs):
    """Solve ``W x = rhs`` for one right-hand side ``rhs`` of shape
    ``(nz, ni)`` given the factors from :func:`factor_block_tridiag`."""
    def fwd(r_p, inp):
        lu_p, piv_p, l_j, b_j = inp
        r_j = b_j - l_j * lu_solve((lu_p, piv_p), r_p)
        return r_j, r_j

    _, r_tail = lax.scan(fwd, rhs[0], (lu[:-1], piv[:-1], sub_d, rhs[1:]))
    r = jnp.concatenate([rhs[0][None], r_tail], axis=0)

    x_last = lu_solve((lu[-1], piv[-1]), r[-1])

    def bwd(x_n, inp):
        lu_j, piv_j, r_j, u_j = inp
        x_j = lu_solve((lu_j, piv_j), r_j - u_j * x_n)
        return x_j, x_j

    _, x_head = lax.scan(bwd, x_last, (lu[:-1], piv[:-1], r[:-1], sup_d), reverse=True)
    return jnp.concatenate([x_head, x_last[None]], axis=0)


def blocks_to_lapack_band(diag, sup_d, sub_d):
    """Test helper: expand block storage into the LAPACK band layout
    ``(3*bw+1, nz*ni)`` with ``bw = 2*ni-1`` used by the banded path, so the
    two kernels can be compared entry by entry.  NumPy, not jitted."""
    import numpy as np
    diag, sup_d, sub_d = (np.asarray(a) for a in (diag, sup_d, sub_d))
    nz, ni, _ = diag.shape
    bw = 2 * ni - 1
    N = nz * ni
    ab = np.zeros((3 * bw + 1, N))
    si, sj = np.mgrid[0:ni, 0:ni]
    for iz in range(nz):
        ab[2 * bw + si - sj, iz * ni + sj] = diag[iz]
    s = np.arange(ni)
    for iz in range(nz - 1):
        # U_iz[s, s] -> A[iz*ni + s, (iz+1)*ni + s]
        ab[2 * bw - ni, (iz + 1) * ni + s] = sup_d[iz]
        # L_{iz+1}[s, s] -> A[(iz+1)*ni + s, iz*ni + s]
        ab[2 * bw + ni, iz * ni + s] = sub_d[iz]
    return ab, bw
