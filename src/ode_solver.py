import numpy as np

from neovulcan_runtime import get_cfg
cfg = get_cfg()
import build_atm
import chemistry_jax as chem_funs
from chemistry_jax import ni
from phy_const import kb, Navo
nz = cfg.atmosphere.nz

from chemistry_jax import neg_achemjac, chem_jac_blocks
from jacobian_jax import _lhs_jac_banded_kernel
from radiative_transfer import make_rt
import jax.numpy as jnp

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class ODESolver:

    # Order of the embedded error estimator used by the adaptive step-size
    # controller.  Ros2 keeps the inherited value of 2; higher-order
    # integrators (e.g. Rodas3) override this in their own class.
    error_order = 2

    def __init__(self):
        self.mtol = cfg.solver.mtol
        self.atol = cfg.solver.atol
        self.non_gas_sp = cfg.condensation.non_gas_sp

        if cfg.condensation.use_condense:
            self.non_gas_sp_index = [species.index(sp) for sp in self.non_gas_sp]
            self.condense_sp_index = [species.index(sp) for sp in cfg.condensation.condense_sp]
            
        self.fix_sp_bot_index = [species.index(sp) for sp in cfg.boundary_conditions.use_fix_sp_bot.keys()]
        self.fix_sp_bot_mix = np.array([cfg.boundary_conditions.use_fix_sp_bot[sp] for sp in cfg.boundary_conditions.use_fix_sp_bot.keys()])
        self.rt = make_rt()

        # Precompute the gas-species mask used by the JAX banded Jacobian.
        # 1.0 for gas species, 0.0 for non-gas — allows ysum to be computed
        # as (y * mask).sum() with no conditional indexing inside the JIT.
        gas_mask = np.ones(ni)
        if cfg.condensation.use_condense:
            gas_mask[:] = 0.0
            # gas_indx is built by build_atm; capture it once per solver instance
            gas_indx = [i for i in range(ni) if species[i] not in self.non_gas_sp]
            gas_mask[gas_indx] = 1.0
        self._gas_mask_jax = jnp.asarray(gas_mask)
        self._use_botflux_flag = jnp.float64(1.0 if cfg.boundary_conditions.use_botflux else 0.0)

        # Cache of JAX-converted atmospheric arrays.  Built lazily on first
        # use and invalidated by Integration.update_mu_dz via
        # invalidate_atm_cache().  Cuts ~0.4 ms/step of jnp.asarray overhead.
        self._atm_jax = None

    # -----------------------------------------------------------------------
    # Private helpers shared by all diffdf variants
    # -----------------------------------------------------------------------

    def _ysum(self, y, atm):
        if cfg.condensation.non_gas_sp:
            return np.sum(y[:, atm.gas_indx], axis=1)
        return np.sum(y, axis=1)

    def _eddy_coeffs(self, ysum, dzi, Kzz, vz):
        """Eddy diffusion + upwind advection tridiagonal coefficients.

        Returns A, B, C of shape (nz,) where:
          dy[j]/dt += A[j]*y[j] + B[j]*y[j+1] + C[j]*y[j-1]
        """
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)

        A[0] = -1./dzi[0] * (Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/2. / ysum[0]
        B[0] =  1./dzi[0] * (Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/2. / ysum[1]
        A[-1] = -1./dzi[-1] * (Kzz[-1]/dzi[-1]) * (ysum[-1]+ysum[-2])/2. / ysum[-1]
        C[-1] =  1./dzi[-1] * (Kzz[-1]/dzi[-1]) * (ysum[-1]+ysum[-2])/2. / ysum[-2]

        A[0]  += -((vz[0]>0)*vz[0])   / dzi[0]
        B[0]  += -((vz[0]<0)*vz[0])   / dzi[0]
        A[-1] +=  ((vz[-1]<0)*vz[-1]) / dzi[-1]
        C[-1] +=  ((vz[-1]>0)*vz[-1]) / dzi[-1]

        j      = np.arange(1, nz-1)
        dz_ave = 0.5*(dzi[j-1] + dzi[j])
        A[j] = -1./dz_ave * (Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2.
                             + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2.) / ysum[j]
        B[j] =  1./dz_ave * Kzz[j]  /dzi[j]   * (ysum[j+1]+ysum[j])/2. / ysum[j+1]
        C[j] =  1./dz_ave * Kzz[j-1]/dzi[j-1] * (ysum[j]+ysum[j-1])/2. / ysum[j-1]
        A[j] += -((vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1]) / dz_ave
        B[j] += -((vz[j]<0)*vz[j])   / dz_ave
        C[j] +=  ((vz[j-1]>0)*vz[j-1]) / dz_ave

        return A, B, C

    def _mol_diff_coeffs(self, ysum, dzi, Dzz, Hpi, Ti, Tco, g, ms, alpha):
        """Molecular diffusion tridiagonal coefficients including thermal diffusion.

        Returns Ai, Bi, Ci of shape (nz, ni).
        """
        Ai, Bi, Ci = [np.zeros((nz, ni)) for _ in range(3)]

        Ai[0] = (-1./dzi[0]*(Dzz[0]/dzi[0])*(ysum[1]+ysum[0])/2./ysum[0]
                 + 1./dzi[0]*Dzz[0]/2.*(-1./Hpi[0] + ms*g[0]/(Navo*kb*Ti[0])
                                         + alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0]))
        Bi[0] = ( 1./dzi[0]*(Dzz[0]/dzi[0])*(ysum[1]+ysum[0])/2./ysum[1]
                 + 1./dzi[0]*Dzz[0]/2.*(-1./Hpi[0] + ms*g[0]/(Navo*kb*Ti[0])
                                         + alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0]))
        Ai[-1] = (-1./dzi[-1]*(Dzz[-1]/dzi[-1])*(ysum[-1]+ysum[-2])/2./ysum[-1]
                  - 1./dzi[-1]*Dzz[-1]/2.*(-1./Hpi[-1] + ms*g[-1]/(Navo*kb*Ti[-1])
                                            + alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1]))
        Ci[-1] = ( 1./dzi[-1]*(Dzz[-1]/dzi[-1])*(ysum[-1]+ysum[-2])/2./ysum[-2]
                  - 1./dzi[-1]*Dzz[-1]/2.*(-1./Hpi[-1] + ms*g[-1]/(Navo*kb*Ti[-1])
                                            + alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1]))

        j    = np.arange(1, nz-1)
        dza  = (0.5*(dzi[j-1]+dzi[j]))[:,None]
        Dj   = Dzz[j];   Dj1  = Dzz[j-1]
        dTj  = (Tco[j+1]-Tco[j])  /dzi[j]
        dTj1 = (Tco[j]  -Tco[j-1])/dzi[j-1]

        Ai[j] = (-1./dza * (Dj /dzi[j][:,None]  *(ysum[j+1]+ysum[j])[:,None]/2.
                            +Dj1/dzi[j-1][:,None]*(ysum[j]+ysum[j-1])[:,None]/2.) / ysum[j][:,None]
                 + 1./(2.*dza) * (Dj  *(-1./Hpi[j][:,None]   + ms*g[j][:,None]  /(Navo*kb*Ti[j][:,None])   + alpha*dTj[:,None] /Ti[j][:,None])
                                 -Dj1 *(-1./Hpi[j-1][:,None] + ms*g[j][:,None]  /(Navo*kb*Ti[j-1][:,None]) + alpha*dTj1[:,None]/Ti[j-1][:,None])))
        Bi[j] = ( 1./dza * Dj/dzi[j][:,None]*(ysum[j+1]+ysum[j])[:,None]/2./ysum[j+1][:,None]
                 + 1./(2.*dza)*Dj*(-1./Hpi[j][:,None] + ms*g[j+1][:,None]/(Navo*kb*Ti[j][:,None]) + alpha*dTj[:,None]/Ti[j][:,None]))
        Ci[j] = ( 1./dza * Dj1/dzi[j-1][:,None]*(ysum[j]+ysum[j-1])[:,None]/2./ysum[j-1][:,None]
                 - 1./(2.*dza)*Dj1*(-1./Hpi[j-1][:,None] + ms*g[j-1][:,None]/(Navo*kb*Ti[j-1][:,None]) + alpha*dTj1[:,None]/Ti[j-1][:,None]))

        return Ai, Bi, Ci

    def _mol_diff_no_thermal_coeffs(self, ysum, dzi, Dzz):
        """Molecular diffusion tridiagonal coefficients without thermal diffusion.

        Used when the thermal/gravity drift is already encoded in an advection
        velocity (vm).  Returns Ai, Bi, Ci of shape (nz, ni).
        """
        Ai, Bi, Ci = [np.zeros((nz, ni)) for _ in range(3)]

        Ai[0]  = -1./dzi[0] *(Dzz[0] /dzi[0]) *(ysum[1]+ysum[0])/2./ysum[0]
        Bi[0]  =  1./dzi[0] *(Dzz[0] /dzi[0]) *(ysum[1]+ysum[0])/2./ysum[1]
        Ai[-1] = -1./dzi[-1]*(Dzz[-1]/dzi[-1])*(ysum[-1]+ysum[-2])/2./ysum[-1]
        Ci[-1] =  1./dzi[-1]*(Dzz[-1]/dzi[-1])*(ysum[-1]+ysum[-2])/2./ysum[-2]

        j    = np.arange(1, nz-1)
        dza  = (0.5*(dzi[j-1]+dzi[j]))[:,None]
        Dj   = Dzz[j];   Dj1 = Dzz[j-1]

        Ai[j] = -1./dza * (Dj /dzi[j][:,None]  *(ysum[j+1]+ysum[j])[:,None]/2.
                           +Dj1/dzi[j-1][:,None]*(ysum[j]+ysum[j-1])[:,None]/2.) / ysum[j][:,None]
        Bi[j] =  1./dza * Dj /dzi[j][:,None]  *(ysum[j+1]+ysum[j])[:,None]/2./ysum[j+1][:,None]
        Ci[j] =  1./dza * Dj1/dzi[j-1][:,None]*(ysum[j]+ysum[j-1])[:,None]/2./ysum[j-1][:,None]

        return Ai, Bi, Ci

    def _upwind_advection(self, dzi, v):
        """Upwind advection coefficients for velocity field v.

        v may be cell-centered ((nz,) or (nz, ni)) or interface-defined
        ((nz-1,) or (nz-1, ni) — e.g. the settling velocity ``atm.vs``).
        Returns (dA, dB, dC) shaped on the cell grid ((nz,) or (nz, ni)) so
        they can be added directly to the caller's tridiagonal arrays.
        """
        out_shape = (nz,) if v.ndim == 1 else (nz, v.shape[1])
        dA = np.zeros(out_shape)
        dB = np.zeros(out_shape)
        dC = np.zeros(out_shape)

        dA[0]  = -((v[0]>0)*v[0])   / dzi[0]
        dB[0]  = -((v[0]<0)*v[0])   / dzi[0]
        dA[-1] =  ((v[-1]<0)*v[-1]) / dzi[-1]
        dC[-1] =  ((v[-1]>0)*v[-1]) / dzi[-1]

        j      = np.arange(1, nz-1)
        dz_ave = 0.5*(dzi[j-1]+dzi[j])
        if v.ndim == 1:
            dA[j] = -((v[j]>0)*v[j] - (v[j-1]<0)*v[j-1]) / dz_ave
            dB[j] = -((v[j]<0)*v[j])   / dz_ave
            dC[j] =  ((v[j-1]>0)*v[j-1]) / dz_ave
        else:
            dza   = dz_ave[:,None]
            dA[j] = -((v[j]>0)*v[j] - (v[j-1]<0)*v[j-1]) / dza
            dB[j] = -((v[j]<0)*v[j])   / dza
            dC[j] =  ((v[j-1]>0)*v[j-1]) / dza

        return dA, dB, dC

    def _apply_tridiag(self, y, A, B, C, Ai=None, Bi=None, Ci=None):
        """Evaluate the tridiagonal operator and return (nz, ni) array.

        A, B, C are scalar (nz,) eddy coefficients.
        Ai, Bi, Ci are optional species-dependent (nz, ni) coefficients.
        """
        if Ai is None:
            tmp0 = A[0]*y[0] + B[0]*y[1]
            tmp1 = np.ndarray.flatten(
                A[1:nz-1, np.newaxis]*y[1:nz-1]
                + B[1:nz-1, np.newaxis]*y[2:nz]
                + C[1:nz-1, np.newaxis]*y[0:nz-2])
            tmp2 = A[-1]*y[-1] + C[-1]*y[-2]
        else:
            tmp0 = (A[0]+Ai[0])*y[0] + (B[0]+Bi[0])*y[1]
            tmp1 = np.ndarray.flatten(
                A[1:nz-1, np.newaxis]*y[1:nz-1]
                + B[1:nz-1, np.newaxis]*y[2:nz]
                + C[1:nz-1, np.newaxis]*y[0:nz-2])
            tmp1 += np.ndarray.flatten(
                Ai[1:nz-1]*y[1:nz-1]
                + Bi[1:nz-1]*y[2:nz]
                + Ci[1:nz-1]*y[0:nz-2])
            tmp2 = (A[-1]+Ai[-1])*y[-1] + (C[-1]+Ci[-1])*y[-2]

        return np.concatenate([tmp0.ravel(), tmp1, tmp2.ravel()]).reshape(nz, ni)

    def _apply_flux_bcs(self, diff, y, atm):
        """Add top/bottom flux boundary contributions in-place."""
        if cfg.boundary_conditions.use_topflux:
            diff[-1] += atm.top_flux / atm.dzi[-1]
        if cfg.boundary_conditions.use_botflux:
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) / atm.dzi[0]
        return diff

    def _subtract_diffusion_to_jac(self, dfdy, A, B, C, Ai=None, Bi=None, Ci=None):
        """Subtract diffusion tridiagonal terms from the dense LHS Jacobian.

        A, B, C are scalar eddy coefficients (nz,).
        Ai, Bi, Ci are optional species-dependent mol-diff coefficients (nz, ni).
        Applies: dfdy[j, j] -= totA[j]; dfdy[j, j+1] -= totB[j]; dfdy[j, j-1] -= totC[j]
        using vectorised scatter over all layer/species index pairs.
        """
        if Ai is not None:
            totA = A[:, None] + Ai   # (nz, ni)
            totB = B[:, None] + Bi
            totC = C[:, None] + Ci
        else:
            totA = np.outer(A, np.ones(ni))
            totB = np.outer(B, np.ones(ni))
            totC = np.outer(C, np.ones(ni))

        idx = np.arange(ni * nz)
        dfdy[idx, idx] -= totA.ravel()

        row_u = np.arange(ni * (nz - 1))
        dfdy[row_u, row_u + ni] -= totB[:nz-1].ravel()

        row_l = np.arange(ni, ni * nz)
        dfdy[row_l, row_l - ni] -= totC[1:].ravel()

    def _diff_esc_to_jac(self, dfdy, y, atm):
        """Apply diffusion-limited escape correction to top-layer diagonal."""
        diff_lim = np.zeros(ni)
        for sp in cfg.boundary_conditions.diff_esc:
            i = species.index(sp)
            if y[-1, i] > 0:
                diff_lim[i] += atm.top_flux[i] / y[-1, i]
        idx_top = np.arange((nz-1)*ni, nz*ni)
        dfdy[idx_top, idx_top] -= diff_lim

    # -----------------------------------------------------------------------

    def diffdf_no_mol(self, y, atm):
        """Eddy diffusion only (no molecular diffusion).

        Zero-flux boundary conditions, non-uniform grid.
        Tridiagonal form: A[j]*y[j] + B[j]*y[j+1] + C[j]*y[j-1]
        """
        ysum = self._ysum(y, atm)
        A, B, C = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        diff = self._apply_tridiag(y, A, B, C)
        return self._apply_flux_bcs(diff, y, atm)
    
    def diffdf(self, y, atm):
        """Eddy + molecular diffusion with thermal diffusion term.

        Zero-flux boundary conditions, non-uniform grid.
        Tridiagonal form: A[j]*y[j] + B[j]*y[j+1] + C[j]*y[j-1]
        """
        ysum = self._ysum(y, atm)
        A, B, C   = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci = self._mol_diff_coeffs(ysum, atm.dzi, atm.Dzz, atm.Hpi,
                                            atm.Ti, atm.Tco, atm.g, atm.ms, atm.alpha)
        diff = self._apply_tridiag(y, A, B, C, Ai, Bi, Ci)
        return self._apply_flux_bcs(diff, y, atm)

    # Clamps for x = log(y) to keep exp(x) inside float64 range while still
    # accepting large transient log-steps.  See `jacobian_jax._X_FLOOR` /
    # `_X_CEIL` — kept in sync.
    X_FLOOR_LOG = -300.0
    X_CEIL_LOG  =  200.0

            
    def diffdf_vm(self, y, atm):
        """Eddy + molecular diffusion (no thermal term) + vm mean-molecular-velocity advection.

        Zero-flux boundary conditions, non-uniform grid.
        """
        ysum = self._ysum(y, atm)
        A, B, C     = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci  = self._mol_diff_no_thermal_coeffs(ysum, atm.dzi, atm.Dzz)
        dAvm, dBvm, dCvm = self._upwind_advection(atm.dzi, atm.vm)
        Ai += dAvm;  Bi += dBvm;  Ci += dCvm
        diff = self._apply_tridiag(y, A, B, C, Ai, Bi, Ci)
        return self._apply_flux_bcs(diff, y, atm)

    def diffdf_settling(self, y, atm):
        """Eddy + molecular diffusion (with thermal term) + particle settling.

        Zero-flux boundary conditions, non-uniform grid.
        """
        ysum = self._ysum(y, atm)
        A, B, C     = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci  = self._mol_diff_coeffs(ysum, atm.dzi, atm.Dzz, atm.Hpi,
                                             atm.Ti, atm.Tco, atm.g, atm.ms, atm.alpha)
        dAvs, dBvs, dCvs = self._upwind_advection(atm.dzi, atm.vs)
        Ai += dAvs;  Bi += dBvs;  Ci += dCvs
        diff = self._apply_tridiag(y, A, B, C, Ai, Bi, Ci)
        return self._apply_flux_bcs(diff, y, atm)

    def diffdf_settling_vm(self, y, atm):
        """Eddy + molecular diffusion (no thermal term) + vm advection + particle settling.

        Zero-flux boundary conditions, non-uniform grid.
        Note: vm is not applied at the bottom boundary (preserved from original).
        """
        ysum = self._ysum(y, atm)
        A, B, C     = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci  = self._mol_diff_no_thermal_coeffs(ysum, atm.dzi, atm.Dzz)
        dAvs, dBvs, dCvs = self._upwind_advection(atm.dzi, atm.vs)
        dAvm, dBvm, dCvm = self._upwind_advection(atm.dzi, atm.vm)
        dAvm[0] = 0;  dBvm[0] = 0   # vm absent at bottom boundary in this variant
        Ai += dAvs + dAvm;  Bi += dBvs + dBvm;  Ci += dCvs + dCvm
        diff = self._apply_tridiag(y, A, B, C, Ai, Bi, Ci)
        return self._apply_flux_bcs(diff, y, atm)
        
        
    def _build_atm_jax_cache(self, atm):
        """Convert atmospheric arrays needed by the JAX Jacobian to JAX once.

        Also pre-computes the variant inputs (vs, vm, Dzz_eff, thermal_flag,
        vm_bot_flag) based on the cfg flags so the per-step call site is
        uniform regardless of which mol-diff/settling/vm variant is active.
        """
        bot_vdep = atm.bot_vdep if cfg.boundary_conditions.use_botflux else np.zeros(ni)

        # Variant-dependent inputs.  See `_lhs_jac_banded_kernel` docstring for
        # the conventions encoded here.
        if cfg.atmosphere.use_moldiff:
            Dzz_eff = atm.Dzz
        else:
            Dzz_eff = np.zeros_like(atm.Dzz)

        if cfg.condensation.use_settling:
            vs_eff = atm.vs
        else:
            vs_eff = np.zeros((nz - 1, ni))

        use_vm_mol = cfg.atmosphere.use_vm_mol
        if use_vm_mol:
            vm_eff = atm.vm
            # In the vm variants the thermal drift is encoded in vm itself,
            # so the mol-diff thermal bracket must be turned off.
            thermal_flag = 0.0
        else:
            vm_eff = np.zeros((nz, ni))
            thermal_flag = 1.0

        # settling_vm only: zero the vm boundary contribution at j=0
        vm_bot_flag = 0.0 if (use_vm_mol and cfg.condensation.use_settling) else 1.0

        self._atm_jax = {
            'M':        jnp.asarray(atm.M),
            'dzi':      jnp.asarray(atm.dzi),
            'Kzz':      jnp.asarray(atm.Kzz),
            'Dzz':      jnp.asarray(Dzz_eff),
            'vz':       jnp.asarray(atm.vz),
            'vs':       jnp.asarray(vs_eff),
            'vm':       jnp.asarray(vm_eff),
            'alpha':    jnp.asarray(atm.alpha),
            'Tco':      jnp.asarray(atm.Tco),
            'ms':       jnp.asarray(atm.ms),
            'g':        jnp.asarray(atm.g),
            'Ti':       jnp.asarray(atm.Ti),
            'Hpi':      jnp.asarray(atm.Hpi),
            'bot_vdep': jnp.asarray(bot_vdep),
            'thermal_flag': jnp.float64(thermal_flag),
            'vm_bot_flag':  jnp.float64(vm_bot_flag),
        }

    def invalidate_atm_cache(self):
        """Drop the cached JAX atm arrays — call after Integration.update_mu_dz."""
        self._atm_jax = None

    def lhs_jac_banded(self, var, atm, c0=None):
        """Build LHS = c0*I - dfdy and return it in **LAPACK band storage**
        (shape ``(3*bw+1, ni*nz)``, ready to hand to ``dgbtrf`` in place).

        The kernel emits the matrix in the compact ``(2*bw+1, ni*nz)`` form
        used by ``scipy.linalg.solve_banded``; we materialise it once into
        the LAPACK layout here so the caller doesn't have to re-allocate and
        re-copy on every step.  Top ``bw`` rows are zero workspace required
        by ``dgbtrf``; the band data occupies rows ``bw`` through ``3*bw``,
        with the main diagonal at row ``2*bw``.

        Returns ``(ab_lapack, bw)``.

        ``c0`` is the diagonal coefficient of the W-method LHS.  If not
        provided, defaults to Ros2's ``1/(r*h)`` with r = 1 + 1/√2.
        Higher-order Rosenbrock methods (e.g. Rodas3) pass their own
        ``c0 = 1/(γ*h)`` consistent with the method's γ.
        """
        from chemistry_jax import k_dict_to_array

        if self._atm_jax is None:
            self._build_atm_jax_cache(atm)
        a = self._atm_jax

        bw = 2 * ni - 1
        k_arr = k_dict_to_array(var.k)
        if c0 is None:
            r  = 1. + 1./np.sqrt(2.)
            c0 = 1. / (r * var.dt)

        ab = _lhs_jac_banded_kernel(
            jnp.asarray(var.y), a['M'], jnp.asarray(k_arr),
            jnp.float64(c0),
            a['dzi'], a['Kzz'], a['Dzz'], a['vz'], a['vs'], a['vm'],
            a['alpha'], a['Tco'],
            a['ms'], a['g'], a['Ti'], a['Hpi'],
            self._gas_mask_jax, a['bot_vdep'],
            self._use_botflux_flag, a['thermal_flag'], a['vm_bot_flag'],
            nz=nz,
        )
        # Materialise into LAPACK layout in one shot: avoids the previous
        # double-copy (JAX→writable numpy, then numpy→ab_lapack inside solver).
        N = ni * nz
        ab_lapack = np.empty((3 * bw + 1, N))
        ab_lapack[:bw] = 0.0
        ab_lapack[bw:] = ab
        return ab_lapack, bw

    def lhs_jac_steady(self, var, atm):
        """Banded -∂F/∂y for the steady-state Newton finisher.

        Same assembly as :meth:`lhs_jac_banded` but with c0 = 0 (no
        identity contribution).  Returns (ab, bw).
        """
        from chemistry_jax import k_dict_to_array

        if self._atm_jax is None:
            self._build_atm_jax_cache(atm)
        a = self._atm_jax

        bw = 2 * ni - 1
        k_arr = k_dict_to_array(var.k)
        ab = _lhs_jac_banded_kernel(
            jnp.asarray(var.y), a['M'], jnp.asarray(k_arr),
            jnp.float64(0.0),
            a['dzi'], a['Kzz'], a['Dzz'], a['vz'], a['vs'], a['vm'],
            a['alpha'], a['Tco'],
            a['ms'], a['g'], a['Ti'], a['Hpi'],
            self._gas_mask_jax, a['bot_vdep'],
            self._use_botflux_flag, a['thermal_flag'], a['vm_bot_flag'],
            nz=nz,
        )
        return np.array(ab), bw

    def steady_newton(self, var, atm):
        """Damped Newton iteration on F(y) = chemdf(y) + diffdf(y) = 0.

        Returns (var, success_flag).  On success ``var.y`` holds the
        converged steady state.  On failure (line search collapses or
        ``newton_max_iter`` reached without converging) ``var`` is
        unchanged and the caller falls back to Rosenbrock time-stepping.

        Only the default eddy + molecular-diffusion path is supported
        (no settling, no vm advection, no use_ion); other configurations
        return ``success=False`` without attempting Newton.
        """
        import scipy.linalg

        if (cfg.condensation.use_settling or cfg.atmosphere.use_vm_mol
                or cfg.photochemistry.use_ion
                or (cfg.condensation.use_condense and cfg.condensation.fix_species)):
            return var, False

        chemdf = chem_funs.chemdf
        diffdf = self.diffdf

        max_iter   = cfg.solver.newton_max_iter
        tol        = cfg.solver.newton_res_tol
        alpha_min  = cfg.solver.newton_alpha_min

        mtol_conv = cfg.solver.mtol_conv

        def scaled_res(y_arr, F_arr):
            """Scaled max-norm residual, mirroring the masking used by
            Integration.conv: species with mixing ratio below mtol_conv or
            number density below atol are excluded — their |F|/|y| is
            dominated by atol floor and is not physically meaningful."""
            ysum = (y_arr * 1).sum(axis=1, keepdims=True)
            ymix = y_arr / np.maximum(ysum, 1.0)
            scale = np.maximum(np.abs(y_arr), self.atol)
            rel = np.abs(F_arr) / scale
            rel = np.where(ymix < mtol_conv, 0.0, rel)
            rel = np.where(y_arr < self.atol, 0.0, rel)
            return float(np.max(rel))

        y = var.y.copy()
        F = chemdf(y, atm.M, var.k) + diffdf(y, atm)
        res  = scaled_res(y, F)
        res0 = res

        for k_iter in range(max_iter):
            if res < tol:
                if cfg.solver.use_print_prog:
                    print(f'  Newton converged in {k_iter} iterations '
                          f'(scaled res {res0:.2e} → {res:.2e})')
                var.y = y
                if cfg.condensation.non_gas_sp:
                    var.ymix = var.y / np.sum(var.y[:, atm.gas_indx], axis=1)[:, np.newaxis]
                else:
                    var.ymix = var.y / np.sum(var.y, axis=1)[:, np.newaxis]
                return var, True

            # Newton direction: solve (-J) dy = F  →  dy = -J^{-1} F
            var.y = y
            ab, bw = self.lhs_jac_steady(var, atm)
            try:
                dy_flat = scipy.linalg.solve_banded((bw, bw), ab, F.flatten())
            except (np.linalg.LinAlgError, ValueError):
                if cfg.solver.use_print_prog:
                    print(f'  Newton solve_banded failed at iter {k_iter}, '
                          f'falling back to Rosenbrock')
                var.y = y
                return var, False
            dy = dy_flat.reshape(y.shape)

            # Damped line search.  Trial is clipped to atol to enforce
            # positivity by projection (rather than rejecting steps).
            # Acceptance: scaled max-norm residual decreased.
            alpha = 1.0
            accepted = False
            while alpha > alpha_min:
                y_trial = np.maximum(y + alpha * dy, self.atol)
                F_trial = chemdf(y_trial, atm.M, var.k) + diffdf(y_trial, atm)
                res_trial = scaled_res(y_trial, F_trial)
                if res_trial < res:
                    y    = y_trial
                    F    = F_trial
                    res  = res_trial
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                if cfg.solver.use_print_prog:
                    print(f'  Newton line search collapsed at iter {k_iter} '
                          f'(scaled res={res:.2e}), falling back to Rosenbrock')
                var.y = y
                return var, False

        if cfg.solver.use_print_prog:
            print(f'  Newton hit max_iter ({max_iter}) without converging (res={res:.2e})')
        var.y = y
        return var, False

    def lhs_jac_banded_numpy(self, var, atm):
        """Legacy NumPy assembly of the banded LHS Jacobian.

        Kept as a reference and as the fallback used by the unit test that
        validates ``_lhs_jac_banded_kernel`` element-wise.  Identical math
        to ``lhs_jac_banded`` but executed entirely in NumPy.
        """
        y = var.y
        if cfg.condensation.use_condense:
            ysum = np.sum(y[:, atm.gas_indx], axis=1)
        else:
            ysum = np.sum(y, axis=1)

        dzi   = atm.dzi
        Kzz   = atm.Kzz
        Dzz   = atm.Dzz
        vz    = atm.vz
        alpha = atm.alpha   # (ni,)
        Tco   = atm.Tco
        ms    = atm.ms      # (ni,)
        g     = atm.g
        Ti    = atm.Ti
        Hpi   = atm.Hpi

        r  = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        bw = 2*ni - 1                          # 63 for ni=32
        ab = np.zeros((2*bw + 1, ni*nz))

        # ------------------------------------------------------------------
        # 1. Chemistry Jacobian blocks → fill into banded matrix
        #    jac[iz, si, sj] = d(dy_si/dt)/d(y_sj) at layer iz  (positive)
        #    banded position: ab[bw + si - sj, iz*ni + sj] = -jac[iz, si, sj]
        # ------------------------------------------------------------------
        jac      = chem_jac_blocks(y, atm.M, var.k)          # (nz, ni, ni)
        si, sj   = np.mgrid[0:ni, 0:ni]                       # (ni, ni) each
        row_chem = bw + si - sj                               # (ni, ni)
        col_chem = np.arange(nz)[:, None, None] * ni + sj    # (nz, ni, ni)
        ab[row_chem[None], col_chem] = -jac                   # broadcast (1,ni,ni)×(nz,ni,ni)

        # ------------------------------------------------------------------
        # 2. Identity: add c0 to main diagonal (banded row bw)
        # ------------------------------------------------------------------
        ab[bw] += c0

        # ------------------------------------------------------------------
        # 3. Diffusion — reshape views of the three active rows
        #
        #    dfdy[j_indx[r], j_indx[c]] -= X  maps to:
        #      r==c   → ab_diag [r]   -= X
        #      c==r+1 → ab_upper[r+1] -= X   (upper off-diagonal in banded)
        #      c==r-1 → ab_lower[r-1] -= X   (lower off-diagonal in banded)
        # ------------------------------------------------------------------
        ab_diag  = ab[bw].reshape(nz, ni)        # (nz, ni) view
        ab_upper = ab[bw - ni].reshape(nz, ni)   # (nz, ni) view
        ab_lower = ab[bw + ni].reshape(nz, ni)   # (nz, ni) view

        # --- middle layers (vectorised over j = 1..nz-2) -----------------
        j      = np.arange(1, nz - 1)            # (nz-2,)
        dz_ave = 0.5*(dzi[j-1] + dzi[j])         # (nz-2,)
        Dj     = Dzz[j]                           # (nz-2,)
        Dj1    = Dzz[j-1]                         # (nz-2,)

        # eddy diffusion (scalar per layer — broadcast over ni species)
        ek_d = (-1./dz_ave * (Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2.
                              + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2.) / ysum[j]
                - ((vz[j] > 0)*vz[j] - (vz[j-1] < 0)*vz[j-1]) / dz_ave)   # (nz-2,)
        ek_u = (1./dz_ave * Kzz[j]/dzi[j] * (ysum[j+1]+ysum[j])/(2.*ysum[j+1])
                - (vz[j] < 0)*vz[j] / dz_ave)                               # (nz-2,)
        ek_l = (1./dz_ave * Kzz[j-1]/dzi[j-1] * (ysum[j-1]+ysum[j])/(2.*ysum[j-1])
                + (vz[j-1] > 0)*vz[j-1] / dz_ave)                          # (nz-2,)

        ab_diag [j]   -= ek_d[:, None]
        ab_upper[j+1] -= ek_u[:, None]
        ab_lower[j-1] -= ek_l[:, None]

        # molecular diffusion
        # Dzz is (nz-1, ni) so Dj=Dzz[j] is (nz-2, ni) — no extra axis needed
        # Hpi, Ti are (nz-1,) so Hpi[j] is (nz-2,) — needs [:, None] to broadcast
        inv_dza  = 1./dz_ave                          # (nz-2,)
        inv_dza2 = inv_dza / 2.                       # (nz-2,)
        dTj      = (Tco[j+1] - Tco[j]) / dzi[j]     # (nz-2,)
        dTj1     = (Tco[j] - Tco[j-1]) / dzi[j-1]  # (nz-2,)

        # Dj is (nz-2, ni); all 1D quantities use [:, None] → (nz-2, 1)
        term_j = Dj * (-1./Hpi[j][:, None]
                       + ms * g[j][:, None] / (Navo*kb*Ti[j][:, None])
                       + alpha * dTj[:, None] / Ti[j][:, None])                 # (nz-2, ni)
        term_j1 = Dj1 * (-1./Hpi[j-1][:, None]
                          + ms * g[j][:, None] / (Navo*kb*Ti[j-1][:, None])
                          + alpha * dTj1[:, None] / Ti[j-1][:, None])           # (nz-2, ni)

        # Dj/dzi[j][:, None]: divide (nz-2,ni) by (nz-2,1) → (nz-2,ni)
        md_d_sc = (-inv_dza[:, None] * (Dj/dzi[j][:, None]*(ysum[j+1]+ysum[j])[:, None]/2.
                                        + Dj1/dzi[j-1][:, None]*(ysum[j-1]+ysum[j])[:, None]/2.)
                   / ysum[j][:, None])                                           # (nz-2, ni)
        md_d = md_d_sc + inv_dza2[:, None] * (term_j - term_j1)                 # (nz-2, ni)

        term_u = Dj * (-1./Hpi[j][:, None]
                       + ms * g[j+1][:, None] / (Navo*kb*Ti[j][:, None])
                       + alpha * dTj[:, None] / Ti[j][:, None])                 # (nz-2, ni)
        md_u = (inv_dza[:, None] * Dj/dzi[j][:, None] * (ysum[j+1]+ysum[j])[:, None]/(2.*ysum[j+1][:, None])
                + inv_dza2[:, None] * term_u)                                    # (nz-2, ni)

        term_l = Dj1 * (-1./Hpi[j-1][:, None]
                         + ms * g[j-1][:, None] / (Navo*kb*Ti[j-1][:, None])
                         + alpha * dTj1[:, None] / Ti[j-1][:, None])            # (nz-2, ni)
        md_l = (inv_dza[:, None] * Dj1/dzi[j-1][:, None] * (ysum[j-1]+ysum[j])[:, None]/(2.*ysum[j-1][:, None])
                - inv_dza2[:, None] * term_l)                                    # (nz-2, ni)

        ab_diag [j]   -= md_d
        ab_upper[j+1] -= md_u
        ab_lower[j-1] -= md_l

        # --- bottom BC (j = 0) -------------------------------------------
        mol_bc0 = (-1./Hpi[0] + ms*g[0]/(Navo*kb*Ti[0])
                   + alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0])                    # (ni,)
        ab_diag [0] -= (-1./dzi[0]*(Kzz[0]/dzi[0])*(ysum[1]+ysum[0])/(2.*ysum[0])
                        - (vz[0] > 0)*vz[0]/dzi[0])
        ab_diag [0] -= (-1./dzi[0]*(Dzz[0]/dzi[0])*(ysum[1]+ysum[0])/(2.*ysum[0])
                        + 1./dzi[0]*Dzz[0]/2.*mol_bc0)
        if cfg.boundary_conditions.use_botflux:
            ab_diag[0] -= -1.*atm.bot_vdep/dzi[0]
        ab_upper[1] -= (1./dzi[0]*(Kzz[0]/dzi[0])*(ysum[1]+ysum[0])/(2.*ysum[1])
                        - (vz[0] < 0)*vz[0]/dzi[0])
        ab_upper[1] -= (1./dzi[0]*(Dzz[0]/dzi[0])*(ysum[1]+ysum[0])/(2.*ysum[1])
                        + 1./dzi[0]*Dzz[0]/2.*mol_bc0)

        # --- top BC (j = nz-1) -------------------------------------------
        mol_bcN = (-1./Hpi[-1] + ms*g[-1]/(Navo*kb*Ti[-1])
                   + alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1])                # (ni,)
        ab_diag [nz-1] -= (-1./dzi[nz-2]*(Kzz[nz-2]/dzi[nz-2])
                            *(ysum[nz-2]+ysum[nz-1])/(2.*ysum[nz-1])
                            + (vz[-1] < 0)*vz[-1]/dzi[-1])
        ab_diag [nz-1] -= (-1./dzi[nz-2]*(Dzz[nz-2]/dzi[nz-2])
                            *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1])
                            - 1./dzi[-1]*Dzz[-1]/2.*mol_bcN)
        ab_lower[nz-2] -= (1./dzi[nz-2]*(Kzz[nz-2]/dzi[nz-2])
                            *(ysum[nz-2]+ysum[nz-1])/(2.*ysum[nz-2])
                            + (vz[-1] > 0)*vz[-1]/dzi[-1])
        ab_lower[nz-2] -= (1./dzi[nz-2]*(Dzz[nz-2]/dzi[nz-2])
                            *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-2])
                            - 1./dzi[-1]*Dzz[-1]/2.*mol_bcN)

        return ab, bw

    def lhs_jac_tot_vm(self, var, atm):
        """LHS Jacobian: eddy + mol diffusion (no thermal) + vm advection."""
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C         = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci      = self._mol_diff_no_thermal_coeffs(ysum, atm.dzi, atm.Dzz)
        dAvm, dBvm, dCvm = self._upwind_advection(atm.dzi, atm.vm)
        Ai += dAvm;  Bi += dBvm;  Ci += dCvm
        self._subtract_diffusion_to_jac(dfdy, A, B, C, Ai, Bi, Ci)

        if cfg.boundary_conditions.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]
        if cfg.boundary_conditions.diff_esc:
            self._diff_esc_to_jac(dfdy, y, atm)

        return dfdy

        
    def lhs_jac_no_mol(self, var, atm):
        """LHS Jacobian: eddy diffusion only (no molecular diffusion)."""
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        self._subtract_diffusion_to_jac(dfdy, A, B, C)

        if cfg.boundary_conditions.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]

        return dfdy
    
    def lhs_jac_fix_all_bot(self, var, atm):
        """LHS Jacobian: eddy + mol diffusion with thermal; fixed bottom BC."""
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C    = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci = self._mol_diff_coeffs(ysum, atm.dzi, atm.Dzz, atm.Hpi,
                                            atm.Ti, atm.Tco, atm.g, atm.ms, atm.alpha)
        self._subtract_diffusion_to_jac(dfdy, A, B, C, Ai, Bi, Ci)

        if cfg.boundary_conditions.diff_esc:
            self._diff_esc_to_jac(dfdy, y, atm)

        # Fixed bottom BC: zero column 0 (removes A[0] diagonal and lower couplings
        # pointing to j=0; the B[0] upper coupling at columns ni..2ni-1 is preserved).
        dfdy[:, :ni] = 0.

        return dfdy
        
    def lhs_jac_no_mol_fix_all_bot(self, var, atm):
        """LHS Jacobian: eddy diffusion only; fixed bottom BC."""
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        self._subtract_diffusion_to_jac(dfdy, A, B, C)

        dfdy[:, :ni] = 0.

        return dfdy

    def lhs_jac_settling(self, var, atm):
        """LHS Jacobian: eddy + mol diffusion with thermal + particle settling."""
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C         = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci      = self._mol_diff_coeffs(ysum, atm.dzi, atm.Dzz, atm.Hpi,
                                                 atm.Ti, atm.Tco, atm.g, atm.ms, atm.alpha)
        dAvs, dBvs, dCvs = self._upwind_advection(atm.dzi, atm.vs)
        Ai += dAvs;  Bi += dBvs;  Ci += dCvs
        self._subtract_diffusion_to_jac(dfdy, A, B, C, Ai, Bi, Ci)

        if cfg.boundary_conditions.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]

        return dfdy
                
    def lhs_jac_settling_vm(self, var, atm):
        """LHS Jacobian: eddy + mol diffusion (no thermal) + particle settling + vm.

        vm is absent at the bottom boundary (preserved from original).
        """
        y = var.y
        ysum = self._ysum(y, atm)
        r = 1. + 1./2.**0.5
        c0 = 1./(r * var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy))

        A, B, C          = self._eddy_coeffs(ysum, atm.dzi, atm.Kzz, atm.vz)
        Ai, Bi, Ci       = self._mol_diff_no_thermal_coeffs(ysum, atm.dzi, atm.Dzz)
        dAvs, dBvs, dCvs = self._upwind_advection(atm.dzi, atm.vs)
        dAvm, dBvm, dCvm = self._upwind_advection(atm.dzi, atm.vm)
        dAvm[0] = 0;  dBvm[0] = 0   # vm absent at bottom boundary
        Ai += dAvs + dAvm;  Bi += dBvs + dBvm;  Ci += dCvs + dCvm
        self._subtract_diffusion_to_jac(dfdy, A, B, C, Ai, Bi, Ci)

        if cfg.boundary_conditions.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]
        if cfg.boundary_conditions.diff_esc:
            self._diff_esc_to_jac(dfdy, y, atm)

        return dfdy
            
        
    def clip(self, var, para, atm):
        pos_cut  = cfg.solver.pos_cut
        nega_cut = cfg.solver.nega_cut
        y, ymix = var.y, var.ymix

        para.small_y += np.abs(np.sum(y[np.logical_and(y<pos_cut, y>=0)]))
        para.nega_y  += np.abs(np.sum(y[np.logical_and(y>nega_cut, y<=0)]))
        y[np.logical_and(y<pos_cut, y>=nega_cut)] = 0.
        y[np.logical_and(ymix<self.mtol, y<0)] = 0.

        var = self.loss(var)

        if cfg.condensation.non_gas_sp:
            var.y, var.ymix = y, var.y / np.sum(var.y[:, atm.gas_indx], axis=1)[:, np.newaxis]
        else:
            var.y, var.ymix = y, y / np.sum(y, axis=1)[:, np.newaxis]

        return var, para
        
    def loss(self, data_var):
        for atom in cfg.network.atom_list:
            if atom not in cfg.boundary_conditions.loss_ex:
                data_var.atom_sum[atom] = np.sum([compo[compo_row.index(species[i])][atom] * data_var.y[:,i] for i in range(ni)])
                data_var.atom_loss[atom] = (data_var.atom_sum[atom] - data_var.atom_ini[atom]) / data_var.atom_ini[atom]
        return data_var
        
    def step_ok(self, var, para):
        loss_eps = cfg.solver.loss_eps
        rtol     = cfg.solver.rtol

        return (np.all(var.y >= 0)
                and np.amax(np.abs(np.fromiter(var.atom_loss.values(), float)
                                   - np.fromiter(var.atom_loss_prev.values(), float))) < loss_eps
                and para.delta <= rtol)
            
    def step_reject(self, var, para):
        rtol = cfg.solver.rtol

        if para.delta > rtol:
            para.delta_count += 1
        elif np.any(var.y < 0):
            para.nega_count += 1
            if cfg.solver.use_print_prog:
                self.print_nega(var, para)
        else:
            para.loss_count += 1
            if cfg.solver.use_print_prog:
                self.print_lossBig(para)

        var = self.reset_y(var)

        # The PI controller's derivative term uses the previously-accepted
        # delta.  After a rejection the local error history is no longer
        # representative — invalidate so the next accepted step falls back
        # to pure I-control.
        para.delta_prev = -1.0

        if var.dt < cfg.solver.dt_min:
            var.dt = cfg.solver.dt_min
            var.y[var.y < 0] = 0.
            print('Keep producing negative values! Clipping negative solutions and moving on!')
            return True

        return False

    def step_size(self, var, para,
                  dt_var_min=cfg.solver.dt_var_min, dt_var_max=cfg.solver.dt_var_max,
                  dt_min=cfg.solver.dt_min, dt_max=cfg.solver.dt_max):
        """Adaptive step-size controller.

        Two controllers, selectable via ``cfg.solver.use_pi_controller``:

        * Legacy I-controller (default; reproduces pre-PI behaviour
          bit-identically):
              h_new = h · 0.9 · (rtol/δ)^(1/p)
        * Gustafsson PI controller (1991; see Hairer-Wanner II §IV.2):
              h_new = h · 0.9 · (rtol/δ)^(α/p) · (δ_prev/δ)^(β/p)
          with α=0.7, β=0.4.  Falls back to I-control on the first step
          and after any rejection — both are flagged by
          ``para.delta_prev < 0`` (set in __init__ and step_reject).

        The integrator's ``error_order`` class attribute selects p
        (2 for Ros2, 3 for Rodas3).
        """
        h     = var.dt
        delta = para.delta
        rtol  = cfg.solver.rtol
        p     = self.error_order

        if delta == 0:
            delta = 0.01 * rtol

        use_pi = (cfg.solver.use_pi_controller
                  and para.delta_prev > 0)
        if use_pi:
            a_over_p = 0.7 / p
            b_over_p = 0.4 / p
            h_factor = (0.9
                        * (rtol / delta) ** a_over_p
                        * (para.delta_prev / delta) ** b_over_p)
        else:
            h_factor = 0.9 * (rtol / delta) ** (1.0 / p)

        h_factor = np.maximum(h_factor, dt_var_min)
        h_factor = np.minimum(h_factor, dt_var_max)
        h *= h_factor
        h = np.maximum(h, dt_min)
        h = np.minimum(h, dt_max)

        var.dt = h
        para.delta_prev = delta
        return var
            
    def reset_y(self, var):
        var.y   = var.y_prev
        var.dt *= cfg.solver.dt_var_min
        return var
        
    def print_nega(self, data_var, data_para):
        nega_i = np.where(data_var.y < 0)
        print('Negative y at time ' + str("{:.2e}".format(data_var.t)) + ' and step: ' + str(data_para.count))
        print('Negative values:' + str(data_var.y[data_var.y < 0]))
        print('from levels: ' + str(nega_i[0]))
        print('species: ' + str([species[_] for _ in nega_i[1]]))
        print('dt= ' + str(data_var.dt))
        print('...reset dt to dt*0.2...')
        print('------------------------------------------------------------------')

    def print_lossBig(self, para):
        print('Element conservation is violated too large')
        print('at step: ' + str(para.count))
        print('------------------------------------------------------------------')
