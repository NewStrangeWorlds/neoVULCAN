import numpy as np
import scipy
from scipy import sparse
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import time, os, pickle
import csv, ast

import vulcan_cfg
try: from PIL import Image
except ImportError:
    try: import Image
    except ImportError: vulcan_cfg.use_PIL = False

import build_atm
import chem_funs
from chem_funs import ni, nr
from phy_const import kb, Navo, hc, ag0
from vulcan_cfg import nz

from chemistry_jax import chemdf, neg_achemjac, chem_jac_blocks
compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class ODESolver(object):
    
    def __init__(self): # do I always need to update var, atm, para ?
        
        self.mtol = vulcan_cfg.mtol
        self.atol = vulcan_cfg.atol
        self.non_gas_sp = vulcan_cfg.non_gas_sp
        
        if vulcan_cfg.use_condense == True:  
            self.non_gas_sp_index = [species.index(sp) for sp in self.non_gas_sp]
            self.condense_sp_index = [species.index(sp) for sp in vulcan_cfg.condense_sp]
            
        self.fix_sp_bot_index = [species.index(sp) for sp in vulcan_cfg.use_fix_sp_bot.keys()]
        self.fix_sp_bot_mix = np.array([vulcan_cfg.use_fix_sp_bot[sp] for sp in vulcan_cfg.use_fix_sp_bot.keys()])

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
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] += -1.*atm.bot_vdep /dzi[0]
        
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
        if vulcan_cfg.use_condense == True:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
            #ysum = np.sum(y, axis=1)
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
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
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
            
        
    def clip(self, var, para, atm, pos_cut = vulcan_cfg.pos_cut, nega_cut = vulcan_cfg.nega_cut):
        '''
        function to clip samll and negative values
        and to calculate the particle loss
        '''
        y, ymix = var.y, var.ymix.copy()
         
        para.small_y += np.abs(np.sum(y[np.logical_and(y<pos_cut, y>=0)]))
        para.nega_y += np.abs(np.sum(y[np.logical_and(y>nega_cut, y<=0)]))
        y[np.logical_and(y<pos_cut, y>=nega_cut)] = 0.
        
        # Also setting y=0 when ymix<mtol
        y[np.logical_and(ymix<self.mtol, y<0)] = 0.
        
        var = self.loss(var)
        
        # store y and ymix
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            var.y, var.ymix = y, var.y/np.vstack(np.sum(var.y[:,atm.gas_indx],axis=1)) 
        else: var.y, var.ymix = y, y/np.vstack(np.sum(y,axis=1))
        # TEST condensation excluding non-gaseous species
        
        return var , para
        
    def loss(self, data_var): 
        
        y = data_var.y
        atom_list = vulcan_cfg.atom_list
        
        # changed atom_tot to dictionary atom_sum
        atom_sum = data_var.atom_sum
        
        for atom in atom_list:
            # data_var.atom_sum[atom] = np.sum([compo[compo_row.index(species[i])][atom] * data_var.y[:,i] for i in range(ni)])
            # TEST V scaling
            if atom not in getattr(vulcan_cfg, 'loss_ex', []): # shami added 2024
                data_var.atom_sum[atom] = np.sum([compo[compo_row.index(species[i])][atom] * data_var.y[:,i] for i in range(ni)]) # *data_var.v_ratio 
                data_var.atom_loss[atom] = (data_var.atom_sum[atom] - data_var.atom_ini[atom])/data_var.atom_ini[atom]

        return data_var
        
    def step_ok(self, var, para, loss_eps = vulcan_cfg.loss_eps, rtol = vulcan_cfg.rtol):
        if np.all(var.y>=0) and np.amax( np.abs( np.fromiter(var.atom_loss.values(),float) - np.fromiter(var.atom_loss_prev.values(),float) ) )<loss_eps and para.delta<=rtol:
            return True
        else:
            return False
            
    def step_reject(self, var, para, loss_eps = vulcan_cfg.loss_eps, rtol = vulcan_cfg.rtol):
        
        if para.delta > rtol: # truncation error larger than the tolerence value
            para.delta_count += 1
            
        elif np.any(var.y < 0):             
            para.nega_count += 1
            if vulcan_cfg.use_print_prog == True:
                self.print_nega(var,para) # print the info for the negative solutions (where y < 0)
            # print input: y, t, count, dt
            

        else: # meaning np.amax( np.abs( np.abs(y_loss) - np.abs(loss_prev) ) )<loss_eps
            para.loss_count +=1
            if vulcan_cfg.use_print_prog == True:
                self.print_lossBig(para)
        
        
        var = self.reset_y(var) # reset y and dt to the values at previous step
        
        if var.dt < vulcan_cfg.dt_min:
            var.dt = vulcan_cfg.dt_min
            var.y[var.y<0] = 0. # clipping of negative values
            
            print ('Keep producing negative values! Clipping negative solutions and moving on!')
            return True
        
        return False
            
    def reset_y(self, var, dt_reduc = vulcan_cfg.dt_var_min):
        '''
        reset y and reduce dt by dt_reduc
        '''
        
        # reset and store y and dt
        var.y = var.y_prev
        var.dt *= dt_reduc
        # var.dt = np.maximum(var.dt, vulcan_cfg.dt_min)

        return var
        
    def print_nega(self, data_var, data_para): 
        
        nega_i = np.where(data_var.y<0)
        print ('Negative y at time ' + str("{:.2e}".format(data_var.t)) + ' and step: ' + str(data_para.count) )
        print ('Negative values:' + str(data_var.y[data_var.y<0]) )
        print ('from levels: ' + str(nega_i[0]) )
        print ('species: ' + str([species[_] for _ in nega_i[1]]) )
        print ('dt= ' + str(data_var.dt))
        print ('...reset dt to dt*0.2...')
        print ('------------------------------------------------------------------')
    
    def print_lossBig(self, para):
        
        print ('Element conservation is violated too large')
        print ('at step: ' + str(para.count))
        print ('------------------------------------------------------------------')
        
    def thomas_vec(a, b, c, d): 
        '''
        Thomas vectorized solver, a b c d refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        d is a matrix
        not used in this current version
        '''
        # number of equations
        nf = len(a) 
        aa, bb, cc, dd = map(np.copy, (a, b, c, d))  
        # d needs to reshape
        dd = dd.reshape(nf,-1)
        #C' and D'
        cp = [cc[0]/bb[0]]; dp = [dd[0]/bb[0]]  
        x = np.zeros((nf, np.shape(dd)[1]))
  
        for i in range(1, nf-1):
            cp.append( cc[i]/(bb[i] - aa[i]*cp[i-1]) ) 
            dp.append( (dd[i] - aa[i]*dp[i-1])/(bb[i] - aa[i]*cp[i-1]) )  
   
        dp.append( (dd[(nf-1)] - aa[(nf-1)]*dp[(nf-1)-1])/(bb[(nf-1)] - aa[(nf-1)]*cp[(nf-1)-1]) ) # nf-1 is the last element
        x[nf-1] = dp[nf-1]/1
        for i in range(nf-2, -1, -1):
            x[i] = dp[i] - cp[i]*x[i+1]
        
        return x
    
    #@njit
    def compute_tau(self, var, atm):
        ''' compute the optical depth '''
        
        # reset to zero
        var.tau.fill(0)
        # absorption species
        absp_sp = set.union(var.photo_sp,var.ion_sp)
            
        for j in range(nz-1,-1,-1):    
            for sp in absp_sp:
                # summing over all T-dependentphoto species
                if sp in vulcan_cfg.T_cross_sp:
                    var.tau[j] += var.y[j,species.index(sp)] * atm.dz[j] * var.cross_T[sp][j]  # 1-D shape of nbins from the j level
                else: # summing over all T-independent photo species    
                    var.tau[j] += var.y[j,species.index(sp)] * atm.dz[j] * var.cross[sp] # only the j-th laye
            
            for sp in vulcan_cfg.scat_sp: # scat_sp are not necessary photo_sp, e.g. He
                var.tau[j] += var.y[j,species.index(sp)] * atm.dz[j] * var.cross_scat[sp]
            # adding the layer above at the end of species loop   
            var.tau[j] += var.tau[j+1]
               
    # Lines like chi = zeta_m**2*tran**2 - zeta_p**2 doing large np 2D array multiplication
    # can be sped up with cython           
    def compute_flux(self, var, atm): # Vectorise this loop!  
        # change it to stagerred grids
        # top: stellar flux
        # bottom BC: zero upcoming flux
        
        # Note!!! Matej's mu is defined in the outgoing hemisphere so his mu<0
        # My cos[sl_angle] is always 0<=mu<=1
        # Converting my mu to Matej's mu (e.g. 45 deg -> 135 deg)
      
        mu_ang = -1.*np.cos(vulcan_cfg.sl_angle)
        edd = vulcan_cfg.edd
        tau = var.tau
        
        # delta_tau (length nz) is used in the transmission function
        delta_tau = tau - np.roll(tau,-1,axis=0) # np.roll(tau,-1,axis=0) are the upper layers
        delta_tau = delta_tau[:-1]
        
        
        # single-scattering albedo
        nbins = len(var.bins)
        tot_abs, tot_scat = np.zeros((nz, nbins)), np.zeros((nz, nbins))
        for sp in var.photo_sp: 
            tot_abs += np.vstack(var.ymix[:,species.index(sp)])*var.cross[sp] # nz * nbins
        for sp in vulcan_cfg.scat_sp: 
            tot_scat += np.vstack(var.ymix[:,species.index(sp)])*var.cross_scat[sp]

        total = tot_abs + tot_scat
        
        w0 = tot_scat  / (tot_abs + tot_scat) # 2D: nz * nbins
        # tot_abs + tot_scat can be zero when certain gas (e.g. H2) does not exist
        
        # Replace nan with zero and inf with very large numbers
        w0 = np.nan_to_num(w0)
        
        # to avoit w0=1
        w0 = np.minimum(w0,1.-1.E-8)

        # sflux: the direct beam; dflux: diffusive flux
        ''' Beer's law for the intensity'''
        var.sflux = var.sflux_top *  np.exp(-1.*tau/np.cos(vulcan_cfg.sl_angle) ) 
        # converting the intensity to flux for the raditive transfer calculation
        dir_flux = var.sflux * np.cos(vulcan_cfg.sl_angle) # need to convert to diffuse flux in the RT definition so it can covert back to total intensity with eps
        
        # scattering
        # the transmission function (length nz)
        if ag0 == 0: # to save memory
            tran = np.exp( -1./edd *(1.- w0)**0.5 * delta_tau ) # 2D: nz * nbins
            zeta_p = 0.5*( 1. + (1.-w0)**0.5 )
            zeta_m = 0.5*( 1. - (1.-w0)**0.5 )
            ll = -1.*w0/( 1./mu_ang**2 -1./edd**2 *(1.-w0) )
            g_p = 0.5*( ll*(1./edd+1./mu_ang) )
            g_m = 0.5*( ll*(1./edd-1./mu_ang) ) 

        else:
            tran = np.exp( -1./edd *( (1.- w0*ag0)*(1.- w0) )**0.5 * delta_tau )
            zeta_p = 0.5*( 1. + ((1.-w0)/(1-w0*ag0))**0.5 )
            zeta_m = 0.5*( 1. - ((1.-w0)/(1-w0*ag0))**0.5 ) 
            ll = ( (1.-w0)*(1-w0*ag0) - 1.)/( 1./mu_ang**2 -1./edd**2 *(1.-w0)*(1-w0*ag0) )
            g_p = 0.5*( ll*(1./edd+1/(mu_ang*(1.-w0*ag0))) + w0*ag0*mu_ang/(1.-w0*ag0)  )
            g_m = 0.5*( ll*(1./edd-1/(mu_ang*(1.-w0*ag0))) - w0*ag0*mu_ang/(1.-w0*ag0)  )
        
        
        # to avoit zero denominator
        ll = np.minimum(ll, 1.e10)
        ll = np.maximum(ll, -1.e10)
        

        # 2D: nz * nbins
        chi = zeta_m**2*tran**2 - zeta_p**2
        xi = zeta_p*zeta_m*(1.-tran**2)
        phi = (zeta_m**2-zeta_p**2)*tran
        
        # 2D: nz * nbins
        i_u = phi*g_p*dir_flux[:-1] - (xi*g_m+chi*g_p)*dir_flux[1:]
        i_d = phi*g_m*dir_flux[1:] - (chi*g_m+xi*g_p)*dir_flux[:-1]
        # sflux[1:] are all the layers above and sflux[:-1] are all the layers abelow
        
        var.zeta_m = zeta_m
        var.zeta_p = zeta_p
        var.tran = tran

        # For testing computating speed
        #starting recording time
        #start_time = timeit.default_timer()

        # propagating downward layer by layer and then upward
        # var.dflux_d and var.dflux_p are defined at the interfaces (staggerred)
        # the rest is defined in the center of the layer
        for j in range(nz-1,-1,-1): # dflux_d goes from the second top interface (nz+1 interfaces) 
            var.dflux_d[j] = 1./chi[j]*(phi[j]*var.dflux_d[j+1] - xi[j]*var.dflux_u[j] + i_d[j]/mu_ang )
        for j in range(1,nz+1):        
            var.dflux_u[j] = 1./chi[j-1]*(phi[j-1]*var.dflux_u[j-1] - xi[j-1]*var.dflux_d[j] + i_u[j-1]/mu_ang )
        

        #print ("time passed...")
        #print (timeit.default_timer() - start_time)
                
        # old
        # # the average intensity (not flux!) of the direct beam
#         ave_int = 0.5*( var.sflux[:-1] + var.sflux[1:])
#         tot_int = (ave_int + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:] + var.dflux_d[1:] + var.dflux_d[:-1]) )/edd
#         # devided by the Eddington coefficient to recover the intensity
        
        
        # the average flux from the direct beam
        # !!! WITHOUT multiplied by the cos zenith angle (flux per unit area perpendicular to the direction of propagationat) !!! 
        ave_dir_flux = 0.5*( var.sflux[:-1] + var.sflux[1:]) 
        # devided by the Eddington coefficient to recover the total intensity (integrated over all directions)
        tot_flux = ave_dir_flux + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:] + var.dflux_d[1:] + var.dflux_d[:-1])/edd 
        
        # For debugging
        #var.ave_int = ave_int
        # var.ll = ll
        # var.chi=chi
        # var.phi=phi
        # var.xi = xi
        # var.i_u = i_u
        # var.i_d = i_d
        # var.w0 = w0
        # var.tot_abs = tot_abs
        # var.tot_scat = tot_scat
        # var.tran = tran
        # var.delta_tau = delta_tau
        # For debugging

        # if np.any(tot_flux< -1.e-20):
        #      print (tot_flux[tot_flux<-1.e-20])
        #      raise IOError ('\nNegative diffusive flux! ')
         
        # store the previous actinic flux into prev_aflux
        var.prev_aflux = np.copy(var.aflux)
        # converting to the actinic flux and storing the current flux
        var.aflux = tot_flux / (hc/var.bins)
        # the change of the actinic flux
        var.aflux_change = np.nanmax( np.abs(var.aflux-var.prev_aflux)[var.aflux>vulcan_cfg.flux_atol]/var.aflux[var.aflux>vulcan_cfg.flux_atol] )
        
        #print ('aflux change: ' + '{:.4E}'.format(var.aflux_change) )
        
    

    # def compute_cross_JT(self, var, atm):
    #     '''
    #     computing T-dependent dissociation cross section based on Tco and stored in the 2D nz*nbins array
    #     only call once at the start
    #     '''
        
        
    
    def compute_J(self, var, atm): # the vectorized version
        '''
        computes photodissociation/photoionization rates; including T-dependent cross sections
        '''
        flux = var.aflux
        
        diss_cross = var.cross_J # use the key (sp, branch index) e.g. ("H2O", 1); 1D array 
        diss_cross_T = var.cross_J_T # 2D array with the shape of nz * bins
            
        bins = var.bins
        n_branch = var.n_branch

        # reset to zeros every time
        var.J_sp = dict([( (sp,bn) , np.zeros(nz)) for sp in var.photo_sp for bn in range(n_branch[sp]+1) ])
         
        for sp in var.photo_sp:
            # shape: flux (nz,nbin) cross (nbin)

            for nbr in range(1, n_branch[sp]+1): # axis=1 is to sum over all wavelength
                if sp in vulcan_cfg.T_cross_sp:
                    var.J_sp[(sp, nbr)] = np.sum( flux[:,:var.sflux_din12_indx] * diss_cross_T[(sp,nbr)][:,:var.sflux_din12_indx] * var.dbin1, axis=1)
                    var.J_sp[(sp, nbr)] -= 0.5* (flux[:,0] * diss_cross_T[(sp,nbr)][:,0] + flux[:,var.sflux_din12_indx-1] * diss_cross_T[(sp,nbr)][:,var.sflux_din12_indx-1]) * var.dbin1
                    var.J_sp[(sp, nbr)] += np.sum( flux[:,var.sflux_din12_indx:] * diss_cross_T[(sp,nbr)][:,var.sflux_din12_indx:] * var.dbin2, axis=1)
                    var.J_sp[(sp, nbr)] -= 0.5* (flux[:,var.sflux_din12_indx] * diss_cross_T[(sp,nbr)][:,var.sflux_din12_indx] + flux[:,-1] * diss_cross_T[(sp,nbr)][:,-1]) * var.dbin2
                    
                else:
                    var.J_sp[(sp, nbr)] = np.sum( flux[:,:var.sflux_din12_indx] * diss_cross[(sp,nbr)][:var.sflux_din12_indx] * var.dbin1, axis=1)
                    var.J_sp[(sp, nbr)] -= 0.5* (flux[:,0] * diss_cross[(sp,nbr)][0] + flux[:,var.sflux_din12_indx-1] * diss_cross[(sp,nbr)][var.sflux_din12_indx-1]) * var.dbin1
                    var.J_sp[(sp, nbr)] += np.sum( flux[:,var.sflux_din12_indx:] * diss_cross[(sp,nbr)][var.sflux_din12_indx:] * var.dbin2, axis=1)
                    var.J_sp[(sp, nbr)] -= 0.5* (flux[:,var.sflux_din12_indx] * diss_cross[(sp,nbr)][var.sflux_din12_indx] + flux[:,-1] * diss_cross[(sp,nbr)][-1]) * var.dbin2
                
                # summing over all branches
                var.J_sp[(sp, 0)] += var.J_sp[(sp, nbr)]
                # incoperating J into rate coefficients
                if var.pho_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
                    var.k[ var.pho_rate_index[(sp, nbr)]  ] = var.J_sp[(sp, nbr)] * vulcan_cfg.f_diurnal # f_diurnal = 0.5 for Earth; = 1 for tidally-loced planets
                                
     
    def compute_Jion(self, var, atm): 
        '''
        compute the photoionization rate
        haven't considered any temperature dependence yet
        '''
        flux = var.aflux
        ion_cross = var.cross_Jion # use the key (sp, br) e.g. ("H2O", 1)
        
        bins = var.bins
        n_branch = var.ion_branch

        # reset to zeros every time
        var.Jion_sp = dict([( (sp,bn) , np.zeros(nz)) for sp in var.ion_sp for bn in range(n_branch[sp]+1) ])

        for sp in var.ion_sp:
            # shape: flux (nz,nbin) cross (nbin)

            # convert to actinic flux *1/(hc/ld)
            for nbr in range(1, n_branch[sp]+1):
                # axis=1 is to sum over all wavelength 
                var.Jion_sp[(sp, nbr)] = np.sum( flux[:,:var.sflux_din12_indx] * ion_cross[(sp,nbr)][:var.sflux_din12_indx] * var.dbin1, axis=1)
                var.Jion_sp[(sp, nbr)] -= 0.5* (flux[:,0] * ion_cross[(sp,nbr)][0]  + flux[:,var.sflux_din12_indx-1] * ion_cross[(sp,nbr)][var.sflux_din12_indx-1]) * var.dbin1
                var.Jion_sp[(sp, nbr)] += np.sum( flux[:,var.sflux_din12_indx:] * ion_cross[(sp,nbr)][var.sflux_din12_indx:] * var.dbin2, axis=1)
                var.Jion_sp[(sp, nbr)] -= 0.5* (flux[:,var.sflux_din12_indx] * ion_cross[(sp,nbr)][var.sflux_din12_indx]  + flux[:,-1] * ion_cross[(sp,nbr)][-1]) * var.dbin2
                
                # 0 is the total dissociation rate
                # summing all branches
 
                var.Jion_sp[(sp, 0)] += var.Jion_sp[(sp, nbr)]
                # incoperating J into rate coefficients
                if var.ion_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
                    var.k[ var.ion_rate_index[(sp, nbr)]  ] = var.Jion_sp[(sp, nbr)] * vulcan_cfg.f_diurnal # f_diurnal = 0.5 for Earth; = 1 for tidally-loced planets        
                # end of the loop: for sp in var.photo_sp:
                     
                    
class Ros2(ODESolver):
    '''
    class inheritance from ODEsolver for 2nd order Rosenbrock solver 
    '''
    def __init__(self):
        #ODESolver.__init__(self)
        super().__init__()
        
           
    def store_bandM(self, a, nb, nn):
        """
        store block-tridiagonal matrix(bandwidth=1) into diagonal ordered form 
        (http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html) 
        a : square block-tridiagonal matirx
        nb: size of the block matrix (number of species)
        nn: number of the block matrices (number of layers)
        """
    
        # band width (treat block-banded as banded matrix)
        bw = 2*nb-1 
        ab = np.zeros((2*bw+1,nb*nn))

        # first 2 columns
        for i in range(0,2*nb):
            ab[-(2*nb+i):,i] = a[0:2*nb+i,i]
    
        # middle
        for i in range(2*nb, nn*nb-2*nb):
            ab[:,i] = a[(i-2*nb+1):(i-2*nb+1)+(2*bw+1),i] 
    
        # last 2 columns
        for ne,i in enumerate(range(nn*nb-2*nb,nn*nb)):
            ab[:(2*bw+1 -ne),i] = a[-(2*bw+1 -ne):,i]
            
        return (ab, bw)

    def solver(self, var, atm, para):
        """
        2nd order Rosenbrock [Verwer et al. 1997] with banded-matrix solver
        with switches to include the molecular diffusion or not
        """
                
        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
        
        if vulcan_cfg.use_vm_mol == False:
            if vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == False:
                diffdf    = self.diffdf
                jac_fn    = self.lhs_jac_banded   # returns (ab, bw) directly
                use_banded = True
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf    = self.diffdf_settling
                jac_fn    = self.lhs_jac_settling
                use_banded = False
            else:
                diffdf    = self.diffdf_no_mol
                jac_fn    = self.lhs_jac_no_mol
                use_banded = False
        else: # vulcan_cfg.use_vm_mol == True:
            if vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == False:
                diffdf    = self.diffdf_vm
                jac_fn    = self.lhs_jac_tot_vm
                use_banded = False
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf    = self.diffdf_settling_vm
                jac_fn    = self.lhs_jac_settling_vm
                use_banded = False
            else:
                diffdf    = self.diffdf_no_mol
                jac_fn    = self.lhs_jac_no_mol
                use_banded = False

        r = 1. + 1./2.**0.5

        df = chemdf(y,M,k).flatten() + diffdf(y, atm).flatten()

        if use_banded:
            lhs_b, bw = jac_fn(var, atm)
            # Fixed species: zero column in banded form, restore diagonal
            if vulcan_cfg.use_condense == True and para.fix_species_start == True:
                for sp in vulcan_cfg.fix_species:
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    lhs_b[:, atm.fix_sp_indx[sp]] = 0.
                    lhs_b[bw, atm.fix_sp_indx[sp]] = 1./(r*h)
            if vulcan_cfg.use_ion == True:
                df[atm.fix_e_indx] = 0
                lhs_b[:, atm.fix_e_indx] = 0.
                lhs_b[bw, atm.fix_e_indx] = 1./(r*h)
        else:
            lhs = jac_fn(var, atm)
            # Fixed species: dense row zeroing
            if vulcan_cfg.use_condense == True and para.fix_species_start == True:
                for sp in vulcan_cfg.fix_species:
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False:
                        pass
                    else:
                        pfix_indx = atm.conden_min_lev[sp]
                        atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                    df[atm.fix_sp_indx[sp]] = 0
                    lhs[atm.fix_sp_indx[sp],:] = 0
                    lhs[atm.fix_sp_indx[sp],atm.fix_sp_indx[sp]] = 1./(r*h)
            if vulcan_cfg.use_ion == True:
                df[atm.fix_e_indx] = 0
                lhs[atm.fix_e_indx,:] = 0
                lhs[atm.fix_e_indx,atm.fix_e_indx] = 1./(r*h)
            lhs_b, bw = self.store_bandM(lhs, ni, nz)

        k1_flat = scipy.linalg.solve_banded((bw,bw),lhs_b,df)
        k1 = k1_flat.reshape(y.shape)
        
        yk2 = y + k1/r
        df = chemdf(yk2,M,k).flatten() + diffdf(yk2, atm).flatten()
        
        # TEST condensation
        # Fixed species
        if vulcan_cfg.use_condense == True and para.fix_species_start == True:
            for sp in vulcan_cfg.fix_species:
                df[atm.fix_sp_indx[sp]] = 0
        if vulcan_cfg.use_ion == True:
            df[atm.fix_e_indx] = 0
            
        rhs = df - 2./(r*h)*k1_flat
        k2 = scipy.linalg.solve_banded((bw,bw),lhs_b,rhs)
        k2 = k2.reshape(y.shape)
        
        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2
        
        ### for Hycean ###
        if getattr(vulcan_cfg, 'use_fix_H2He', False) and 'H2' not in vulcan_cfg.use_fix_sp_bot and var.t > 1e6:
            vulcan_cfg.use_fix_sp_bot['H2'] = var.ymix[0,species.index('H2')]
            vulcan_cfg.use_fix_sp_bot['He'] = var.ymix[0,species.index('He')]
            print ("After 1e6 sec, H2 and He are fixed at " + str((var.ymix[0,species.index('H2')], var.ymix[0,species.index('He')])))  
            
            self.fix_sp_bot_index = [species.index(sp) for sp in vulcan_cfg.use_fix_sp_bot.keys()]
            self.fix_sp_bot_mix = np.array([vulcan_cfg.use_fix_sp_bot[sp] for sp in vulcan_cfg.use_fix_sp_bot.keys()])
        ### for Hycean ###
        
        # setting particles on the surace = 0
        if vulcan_cfg.use_fix_sp_bot: # if use_fix_sp_bot = {} (empty), it returns false
            sol[0,self.fix_sp_bot_index] = self.fix_sp_bot_mix*atm.n_0[0]
                
        delta = np.abs(sol-yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0
                
        # neglecting the errors at the surface
        if vulcan_cfg.use_botflux == True or vulcan_cfg.use_fix_sp_bot: delta[0] = 0
        
        # TEST condensation 2022
        if vulcan_cfg.use_condense == True:
            delta[:,self.non_gas_sp_index] = 0
            delta[:,self.condense_sp_index] = 0

            if para.fix_species_start == True:

                for sp in vulcan_cfg.fix_species: 
                    if vulcan_cfg.fix_species_from_coldtrap_lev == False: # if Ptop is not specified, fix the whole column # TEST2022
                        sol[:,species.index(sp)] = var.fix_y[sp].copy() 
                    else:
                        #pfix_indx = min( range(len(atm.pco)), key=lambda i: abs(atm.pco[i]- vulcan_cfg.fix_species_Ptop[0] ))
                        pfix_indx = atm.conden_min_lev[sp]
                        sol[:pfix_indx,species.index(sp)] = var.fix_y[sp].copy()[:pfix_indx]

                    delta[:,species.index(sp)] = 0

            # Recorde the condensing levels TEST 2022 # do we need this?
            # for sp in ['H2O']:
            #     conden_status = sol[:,species.index(sp)] >= atm.n_0 * atm.sat_mix[sp]*0.99
            #     atm.conden_status = conden_status
            # Recorde the condensing levels TEST 2022

        if vulcan_cfg.use_print_delta == True and para.count % vulcan_cfg.print_prog_num==0:
            max_indx = np.nanargmax(delta/sol, axis=1)
            max_lev_indx = np.nanargmax(delta/sol)
            print ('Largest delta (truncation error) from nz = ' + str(int(max_lev_indx/ni) ) )
            print ( np.array(species)[max_indx] )
            print ('Largest delta (truncation error) from ' + species[max_indx%ni] + " at nz = "   + str(int(max_indx/ni) ) ) 

        delta = np.amax( delta[sol>0]/sol[sol>0] )
        
        var.y = sol
        
        # # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            var.ymix = var.y/np.vstack(np.sum(var.y[:,atm.gas_indx],axis=1))
        else:
            var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
        # TEST condensation excluding non-gaseous species
        
        para.delta = delta    
        
        # use charge balance to obtain the number density of electrons (such that [ions] = [e])
        if vulcan_cfg.use_ion == True: 
            # clear e
            var.y[:,species.index('e')] = 0
            # set e such that the net chare is zero
            for sp in var.charge_list:
                var.y[:,species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:,species.index(sp)]
        
        
        return var, para
        
    def solver_fix_all_bot(self, var, atm, para):
        """
        2nd order Rosenbrock [Verwer et al. 1997] with banded-matrix solver
        with switches to include the molecular diffusion or not
        """

        y, ymix, h, k = var.y, var.ymix, var.dt, var.k
        M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
        
        # store the fixed bottom level
        bottom = np.copy(ymix[0])
        
        if vulcan_cfg.use_moldiff == True:
            diffdf = self.diffdf
            jac_tot = self.lhs_jac_fix_all_bot
        else:
            diffdf = self.diffdf_no_mol
            jac_tot = self.lhs_jac_no_mol_fix_all_bot
    
        r = 1. + 1./2.**0.5

        df = chemdf(y,M,k).flatten() + diffdf(y, atm).flatten()
        lhs = jac_tot(var, atm)
        
        lhs_b, bw = self.store_bandM(lhs,ni,nz)
        k1_flat = scipy.linalg.solve_banded((bw,bw),lhs_b,df)
        
        k1 = k1_flat.reshape(y.shape)
        
        yk2 = y + k1/r
        df = chemdf(yk2,M,k).flatten() + diffdf(yk2, atm).flatten()
        
        rhs = df - 2./(r*h)*k1_flat
        k2 = scipy.linalg.solve_banded((bw,bw),lhs_b,rhs)
        k2 = k2.reshape(y.shape)
        
        sol = y + 3./(2.*r)*k1 + 1/(2.*r)*k2  
        
        # fixed the bottom layer to yini (in chemical EQ)
        sol[0] = bottom*atm.n_0[0] 
                    
        delta = np.abs(sol-yk2)
        delta[ymix < self.mtol] = 0
        delta[sol < self.atol] = 0
        
        delta = np.amax( delta[sol>0]/sol[sol>0] )

        var.y = sol
        
        # # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            var.ymix = var.y/np.vstack(np.sum(var.y[:,atm.gas_indx],axis=1))
        else:
            var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
        # TEST condensation excluding non-gaseous species
        
        para.delta = delta    
        
        # use charge balance to obtain the number density of electrons (such that [ions] = [e])
        if vulcan_cfg.use_ion == True: 
            # clear e
            var.y[:,species.index('e')] = 0
            # set e such that the net chare is zero
            for sp in var.charge_list:
                var.y[:,species.index('e')] -= compo[compo_row.index(sp)]['e'] * var.y[:,species.index(sp)]

        return var, para   
      
    
    def naming_solver(self, para):
        
        # if vulcan_cfg.use_fix_all_bot == True:
        #     if vulcan_cfg.use_moldiff == True: print ('Use fixed bottom BC and molecular diffusion.')
        #     else: print ('Use fixed bottom BC and No molecular diffusion.')
        #     para.solver_str = 'solver_fix_all_bot'
            
        #else:
        if vulcan_cfg.use_moldiff == True: print ('Include molecular diffusion.')
        else: print ('No molecular diffusion.')
        para.solver_str = 'solver'
        
        
    def one_step(self, var, atm, para):

        while True:
            
           var, para =  getattr(self, para.solver_str)(var, atm, para)
           
           # clipping small negative values and also calculating atomic loss (atom_loss)  
           var , para = self.clip(var, para, atm) 
            
           if self.step_ok(var, para): break
           elif self.step_reject(var, para): break # giving up and moving on
                  
        return var, para                    
        
    def step_size(self, var, para, dt_var_min = vulcan_cfg.dt_var_min, dt_var_max = vulcan_cfg.dt_var_max, dt_min = vulcan_cfg.dt_min, dt_max = vulcan_cfg.dt_max):  
        """
        step-size control by delta(truncation error) for the Rosenbrock method
        """
        h = var.dt
        delta = para.delta
        rtol = vulcan_cfg.rtol
               
        if delta==0: delta = 0.01*rtol
        h_factor = 0.9*(rtol/delta)**0.5 # 0.9 is simply a safety factor
        h_factor = np.maximum(h_factor, dt_var_min)    
        h_factor = np.minimum(h_factor, dt_var_max)    
        
        h *= h_factor
        h = np.maximum(h, dt_min)
        h = np.minimum(h, dt_max)
        
        # store the adopted dt
        var.dt = h
        
        return var
            
    
