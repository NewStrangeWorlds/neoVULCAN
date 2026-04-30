import numpy as np

import vulcan_cfg
import build_atm
import chem_funs
from chem_funs import ni
from phy_const import kb, Navo
from vulcan_cfg import nz

from chemistry_jax import neg_achemjac, chem_jac_blocks
from radiative_transfer import TwoStreamRT

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class ODESolver:

    def __init__(self):
        self.mtol = vulcan_cfg.mtol
        self.atol = vulcan_cfg.atol
        self.non_gas_sp = vulcan_cfg.non_gas_sp

        if vulcan_cfg.use_condense:
            self.non_gas_sp_index = [species.index(sp) for sp in self.non_gas_sp]
            self.condense_sp_index = [species.index(sp) for sp in vulcan_cfg.condense_sp]
            
        self.fix_sp_bot_index = [species.index(sp) for sp in vulcan_cfg.use_fix_sp_bot.keys()]
        self.fix_sp_bot_mix = np.array([vulcan_cfg.use_fix_sp_bot[sp] for sp in vulcan_cfg.use_fix_sp_bot.keys()])
        self.rt = TwoStreamRT()

    # -----------------------------------------------------------------------
    # Private helpers shared by all diffdf variants
    # -----------------------------------------------------------------------

    def _ysum(self, y, atm):
        if vulcan_cfg.non_gas_sp:
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

        v may be 1-D (nz,) for a scalar velocity or 2-D (nz, ni) for a
        species-dependent velocity.  Returns (dA, dB, dC) with the same shape
        as v, to be added to the caller's tridiagonal arrays.
        """
        dA = np.zeros_like(v)
        dB = np.zeros_like(v)
        dC = np.zeros_like(v)

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
                np.vstack(A[1:nz-1])*y[1:nz-1]
                + np.vstack(B[1:nz-1])*y[2:nz]
                + np.vstack(C[1:nz-1])*y[0:nz-2])
            tmp2 = A[-1]*y[-1] + C[-1]*y[-2]
        else:
            tmp0 = (A[0]+Ai[0])*y[0] + (B[0]+Bi[0])*y[1]
            tmp1 = np.ndarray.flatten(
                np.vstack(A[1:nz-1])*y[1:nz-1]
                + np.vstack(B[1:nz-1])*y[2:nz]
                + np.vstack(C[1:nz-1])*y[0:nz-2])
            tmp1 += np.ndarray.flatten(
                Ai[1:nz-1]*y[1:nz-1]
                + Bi[1:nz-1]*y[2:nz]
                + Ci[1:nz-1]*y[0:nz-2])
            tmp2 = (A[-1]+Ai[-1])*y[-1] + (C[-1]+Ci[-1])*y[-2]

        return np.concatenate([tmp0.ravel(), tmp1, tmp2.ravel()]).reshape(nz, ni)

    def _apply_flux_bcs(self, diff, y, atm):
        """Add top/bottom flux boundary contributions in-place."""
        if vulcan_cfg.use_topflux:
            diff[-1] += atm.top_flux / atm.dzi[-1]
        if vulcan_cfg.use_botflux:
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
        for sp in vulcan_cfg.diff_esc:
            if y[-1, species.index(sp)] > 0:
                diff_lim[species.index(sp)] += (atm.top_flux[species.index(sp)]
                                                 / y[-1, species.index(sp)])
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
        
        
    def jac_tot(self, var, atm):
        """
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion)
        zero-flux BC:  1st derivitive of y is zero
        """
        
        y = var.y.copy()
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        Dzz = atm.Dzz.copy()
        vz = atm.vz.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        mu, ms = atm.mu.copy(),  atm.ms.copy()
        g = atm.g
        
        # define T_1/2 for the molecular diffusion
        #Ti = 0.5*(Tco + np.roll(Tco,-1))
        #Ti = Ti[:-1]
        
        Ti = atm.Ti.copy()
        Hpi = atm.Hpi.copy()
        
        
        dfdy = achemjac(y, atm.M, var.k)
        j_indx = []
        
        for j in range(nz):
            j_indx.append( np.arange(j*ni,j*ni+ni) )

        for j in range(1,nz-1): 
            # excluding the buttom and the top cell
            # at j level consists of ni species
            dz_ave = 0.5*(dzi[j-1] + dzi[j]) 
            dfdy[j_indx[j], j_indx[j]] +=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave  
            dfdy[j_indx[j], j_indx[j+1]] += 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
            dfdy[j_indx[j], j_indx[j-1]] += 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave
            
            # [j_indx[j], j_indx[j]] has size ni*ni
            dfdy[j_indx[j], j_indx[j]] +=  -1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j]\
            +1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g[j]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
            - Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) )
            dfdy[j_indx[j], j_indx[j+1]] += 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
            +1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g[j+1]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) 
            dfdy[j_indx[j], j_indx[j-1]] += 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
            -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j-1]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )

              
        dfdy[j_indx[0], j_indx[0]] += -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[0]] += -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] ) 
        # deposition velocity
        if vulcan_cfg.use_botflux: dfdy[j_indx[0], j_indx[0]] += -1.*atm.bot_vdep /dzi[0]
        
        dfdy[j_indx[0], j_indx[1]] += 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0] 
        dfdy[j_indx[0], j_indx[1]] += 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )

        dfdy[j_indx[nz-1], j_indx[nz-1]] += -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1] 
        dfdy[j_indx[nz-1], j_indx[nz-1]] += -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
        - 1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] += 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] += 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
                -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )

        return dfdy

    def lhs_jac_tot(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion)
        zero-flux BC:  1st derivitive of y is zero
        """
        y = var.y.copy()
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.use_condense:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else:
            ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        Dzz = atm.Dzz.copy()
        vz = atm.vz.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        mu, ms = atm.mu.copy(),  atm.ms.copy()
        g = atm.g

        Ti = atm.Ti.copy()
        Hpi = atm.Hpi.copy()

        # c0 = 1./(r*h) where r = 1. + 1./2.**0.5
        r = 1. + 1./2.**0.5
        c0 = 1./(r*var.dt)
        dfdy = neg_achemjac(y, atm.M, var.k)
        np.fill_diagonal(dfdy, c0 + np.diag(dfdy)) 
        j_indx = []
        
        for j in range(nz):
            j_indx.append( np.arange(j*ni,j*ni+ni) )

        for j in range(1,nz-1):
            # excluding the buttom and the top cell
            # at j level consists of ni species
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
            dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave

            # [j_indx[j], j_indx[j]] has size ni*ni
            dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j]\
            +1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g[j]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
            - Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) )
            dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
            +1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g[j+1]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )
            dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
            -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j-1]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )
    
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] ) 
        # deposition velocity
        if vulcan_cfg.use_botflux: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
        - 1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
                -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )

        return dfdy

    def lhs_jac_banded(self, var, atm):
        """Build LHS = 1/(r*h)*I - dfdy directly in scipy banded format.

        Equivalent to lhs_jac_tot but avoids the 118 MB dense block-diagonal
        matrix.  Returns (ab, bw) ready for scipy.linalg.solve_banded.

        Banded mapping:
          ab[bw + (i-j), j] = dense[i, j]   (scipy convention)
        where rows are layer-major: index iz*ni+sp.

        Three banded rows are active for diffusion:
          row bw      — same-layer, same-species (diagonal + eddy + mol)
          row bw-ni   — upper off-diagonal  (layer j coupled to column j+1)
          row bw+ni   — lower off-diagonal  (layer j coupled to column j-1)
        Chemistry fills all rows bw±(0..ni-1) via the block fill.
        """
        y = var.y.copy()
        if vulcan_cfg.use_condense:
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
        if vulcan_cfg.use_botflux:
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

        if vulcan_cfg.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]
        if vulcan_cfg.diff_esc:
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

        if vulcan_cfg.use_botflux:
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

        if vulcan_cfg.diff_esc:
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

        if vulcan_cfg.use_botflux:
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

        if vulcan_cfg.use_botflux:
            idx0 = np.arange(ni)
            dfdy[idx0, idx0] -= -atm.bot_vdep / atm.dzi[0]
        if vulcan_cfg.diff_esc:
            self._diff_esc_to_jac(dfdy, y, atm)

        return dfdy
            
        
    def clip(self, var, para, atm):
        pos_cut  = vulcan_cfg.pos_cut
        nega_cut = vulcan_cfg.nega_cut
        y, ymix = var.y, var.ymix.copy()

        para.small_y += np.abs(np.sum(y[np.logical_and(y<pos_cut, y>=0)]))
        para.nega_y  += np.abs(np.sum(y[np.logical_and(y>nega_cut, y<=0)]))
        y[np.logical_and(y<pos_cut, y>=nega_cut)] = 0.
        y[np.logical_and(ymix<self.mtol, y<0)] = 0.

        var = self.loss(var)

        if vulcan_cfg.non_gas_sp:
            var.y, var.ymix = y, var.y / np.vstack(np.sum(var.y[:, atm.gas_indx], axis=1))
        else:
            var.y, var.ymix = y, y / np.vstack(np.sum(y, axis=1))

        return var, para
        
    def loss(self, data_var):
        for atom in vulcan_cfg.atom_list:
            if atom not in getattr(vulcan_cfg, 'loss_ex', []):
                data_var.atom_sum[atom] = np.sum([compo[compo_row.index(species[i])][atom] * data_var.y[:,i] for i in range(ni)])
                data_var.atom_loss[atom] = (data_var.atom_sum[atom] - data_var.atom_ini[atom]) / data_var.atom_ini[atom]
        return data_var
        
    def step_ok(self, var, para):
        loss_eps = vulcan_cfg.loss_eps
        rtol     = vulcan_cfg.rtol

        return (np.all(var.y >= 0)
                and np.amax(np.abs(np.fromiter(var.atom_loss.values(), float)
                                   - np.fromiter(var.atom_loss_prev.values(), float))) < loss_eps
                and para.delta <= rtol)
            
    def step_reject(self, var, para):
        rtol = vulcan_cfg.rtol

        if para.delta > rtol:
            para.delta_count += 1
        elif np.any(var.y < 0):
            para.nega_count += 1
            if vulcan_cfg.use_print_prog:
                self.print_nega(var, para)
        else:
            para.loss_count += 1
            if vulcan_cfg.use_print_prog:
                self.print_lossBig(para)

        var = self.reset_y(var)

        if var.dt < vulcan_cfg.dt_min:
            var.dt = vulcan_cfg.dt_min
            var.y[var.y < 0] = 0.
            print('Keep producing negative values! Clipping negative solutions and moving on!')
            return True

        return False
            
    def reset_y(self, var):
        var.y   = var.y_prev
        var.dt *= vulcan_cfg.dt_var_min
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
