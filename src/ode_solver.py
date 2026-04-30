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

chemdf = chem_funs.chemdf
neg_achemjac = chem_funs.neg_symjac
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
  
    def diffdf_no_mol(self, y, atm): 
        """
        function of eddy diffusion without molecular diffusion, with zero-flux boundary conditions and non-uniform grids (dzi)
        in the form of Aj*y_j + Bj+1*y_j+1 + Cj-1*y_j-1
        """
        y = y.copy()
        # TEST excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST excluding non-gaseous species
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)

        A[0] = -1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]     
        B[0] = 1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] 
        C[0] = 0 
        A[nz-1] = -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] 
        B[nz-1] = 0 
        C[nz-1] = 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] 
        
        # vertical adection with zero-flux B.C. 
        A[0] += -( (vz[0]>0)*vz[0] )/dzi[0]
        B[0] += -( (vz[0]<0)*vz[0] )/dzi[0]
        A[-1] += ( (vz[-1]<0)*vz[-1] )/dzi[-1]
        C[-1] += ( (vz[-1]>0)*vz[-1] )/dzi[-1]
        # vertical adection
        
        for j in range(1,nz-1):  
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            A[j] = -2./(dzi[j-1] + dzi[j])* ( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            B[j] = 2./(dzi[j-1] + dzi[j])*Kzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            C[j] = 2./(dzi[j-1] + dzi[j])*Kzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # vertical adection
            A[j] += -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            B[j] += -( (vz[j]<0)*vz[j] )/dz_ave
            C[j] += ( (vz[j-1]>0)*vz[j-1] )/dz_ave
            # vertical adection
            
        tmp0 = A[0]*y[0] + B[0]*y[1]
        tmp1 = np.ndarray.flatten( (np.vstack(A[1:nz-1])*y[1:(nz-1)] + np.vstack(B[1:nz-1])*y[1+1:(nz-1)+1] + np.vstack(C[1:nz-1])*y[1-1:(nz-1)-1]) )
        tmp2 = (A[nz-1]*y[nz-1] +C[nz-1]*y[nz-2]) 
        diff = np.append(np.append(tmp0, tmp1), tmp2)
        diff = diff.reshape(nz,ni)
        
        if vulcan_cfg.use_topflux == True:
            # Don't forget dz!!! -d phi/ dz
            ### the const flux has no contribution to the jacobian ### 
            diff[-1] += atm.top_flux /dzi[-1]
        if vulcan_cfg.use_botflux == True:
            ### the deposition term needs to be included in the jacobian!!!   
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) /dzi[0]
        return diff
    
    def diffdf(self, y, atm): 
        """
        function of eddy diffusion including molecular diffusion, with zero-flux boundary conditions and non-uniform grids (dzi)
        in the form of Aj*y_j + Bj+1*y_j+1 + Cj-1*y_j-1
        """
        
        y = y.copy()
        
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
    
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        Dzz = atm.Dzz.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        ms = atm.ms.copy()
        Hp = atm.Hp.copy()
        g = atm.g
        Ti = atm.Ti
        Hpi = atm.Hpi
        
        # # define T_1/2 for the molecular diffusion
        # Ti = 0.5*(Tco + np.roll(Tco,-1))
        # Ti = Ti[:-1]
        # Hpi = 0.5*(Hp + np.roll(Hp,-1))
        # Hpi = Hpi[:-1]
        # # store Ti and Hpi
        # atm.Ti = Ti
        # atm.Hpi = Hpi
        
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)
        Ai, Bi, Ci = [ np.zeros((nz,ni)) for i in range(3)]
        
        A[0] = -1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]     
        B[0] = 1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] 
        C[0] = 0 
        A[nz-1] = -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] 
        B[nz-1] = 0 
        C[nz-1] = 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] 
        
        # vertical adection (with closed B.C.) 
        A[0] += -( (vz[0]>0)*vz[0] )/dzi[0]
        B[0] += -( (vz[0]<0)*vz[0] )/dzi[0]
        A[-1] += ( (vz[-1]<0)*vz[-1] )/dzi[-1]
        C[-1] += ( (vz[-1]>0)*vz[-1] )/dzi[-1]
        # vertical adection
         
        # shape of ni-long 1D array
        Ai[0] = -1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0] +\
        1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )  
        Bi[0] = 1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] +\
        1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )
        Ci[0] = 0 
        Ai[nz-1] = -1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] \
        -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
        Bi[nz-1] = 0
        Ci[nz-1] = 1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] \
        -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
        
        for j in range(1,nz-1):
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            A[j] = -1./dz_ave * ( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            B[j] = 1./dz_ave * Kzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            C[j] = 1./dz_ave * Kzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # vertical adection
            A[j] += -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            B[j] += -( (vz[j]<0)*vz[j] )/dz_ave
            C[j] += ( (vz[j-1]>0)*vz[j-1] )/dz_ave
            # vertical adection
            
            # Ai in the shape of nz*ni and Ai[j] in the shape of ni 
            Ai[j] = -1./dz_ave * ( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            Bi[j] = 1./dz_ave * Dzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            Ci[j] = 1./dz_ave * Dzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            Ai[j] += 1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g[j]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
            - Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j]/(Navo*kb*Ti[j-1])+ alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) ) #/ysum[j]
            Bi[j] += 1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g[j+1]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )
            Ci[j] += -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j-1]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )
 
        tmp0 = (A[0] + Ai[0])*y[0] + (B[0] + Bi[0])*y[1] # shape of ni-long 1D array  
        tmp1 = np.ndarray.flatten( (np.vstack(A[1:nz-1])*y[1:(nz-1)] + np.vstack(B[1:nz-1])*y[1+1:(nz-1)+1] + np.vstack(C[1:nz-1])*y[1-1:(nz-1)-1]) ) 
        tmp1 += np.ndarray.flatten( Ai[1:nz-1]*y[1:(nz-1)] + Bi[1:nz-1]*y[1+1:(nz-1)+1] + Ci[1:nz-1]*y[1-1:(nz-1)-1] ) # shape of (nz-2,ni)
        tmp2 = (A[nz-1] + Ai[nz-1])*y[nz-1] + (C[nz-1] + Ci[nz-1])*y[nz-2]
        diff = np.append(np.append(tmp0, tmp1), tmp2)
        diff = diff.reshape(nz,ni)

        if vulcan_cfg.use_topflux == True:
            # Don't forget dz!!! -d phi/ dz
            ### the const flux has no contribution to the jacobian ### 
            diff[-1] += atm.top_flux /dzi[-1]
        if vulcan_cfg.use_botflux == True:
            ### the deposition term needs to be included in the jacobian!!!   
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) /dzi[0]
        
        return diff
            
    def diffdf_vm(self, y, atm): 
        """
        function of eddy diffusion including molecular diffusion, with zero-flux boundary conditions and non-uniform grids (dzi)
        in the form of Aj*y_j + Bj+1*y_j+1 + Cj-1*y_j-1
        inc. vm from molecular diffusion
        """
        
        y = y.copy()
        
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
    
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        Dzz = atm.Dzz.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        ms = atm.ms.copy()
        Hp = atm.Hp.copy()
        g = atm.g
        Ti = atm.Ti
        Hpi = atm.Hpi
        
        vm = atm.vm
        
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)
        Ai, Bi, Ci = [ np.zeros((nz,ni)) for i in range(3)]
        
        A[0] = -1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]     
        B[0] = 1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] 
        C[0] = 0 
        A[nz-1] = -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] 
        B[nz-1] = 0 
        C[nz-1] = 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] 
        
        # vertical adection (with closed B.C.) 
        A[0] += -( (vz[0]>0)*vz[0] )/dzi[0]
        B[0] += -( (vz[0]<0)*vz[0] )/dzi[0]
        A[-1] += ( (vz[-1]<0)*vz[-1] )/dzi[-1]
        C[-1] += ( (vz[-1]>0)*vz[-1] )/dzi[-1]
        # vertical adection
         
        # shape of ni-long 1D array
        Ai[0] = -1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]   -( (vm[0]>0)*vm[0] )/dzi[0]   
        Bi[0] = 1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1]    -( (vm[0]<0)*vm[0] )/dzi[0]
        Ci[0] = 0 
        Ai[nz-1] = -1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] \
        +( (vm[-1]<0)*vm[-1] )/dzi[-1] 
        Bi[nz-1] = 0
        Ci[nz-1] = 1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] \
        +( (vm[-1]>0)*vm[-1] )/dzi[-1]
        
        for j in range(1,nz-1):
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            A[j] = -1./dz_ave * ( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            B[j] = 1./dz_ave * Kzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            C[j] = 1./dz_ave * Kzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # vertical adection
            A[j] += -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            B[j] += -( (vz[j]<0)*vz[j] )/dz_ave
            C[j] += ( (vz[j-1]>0)*vz[j-1] )/dz_ave
            # vertical adection
            
            # Ai in the shape of nz*ni and Ai[j] in the shape of ni 
            # diffusion component
            Ai[j] = -1./dz_ave * ( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            Bi[j] = 1./dz_ave * Dzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            Ci[j] = 1./dz_ave * Dzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # advective component using upwind (inc. from Dzz and from vs)
            Ai[j] += -( (vm[j]>0)*vm[j] - (vm[j-1]<0)*vm[j-1] )/dz_ave 
            Bi[j] += -( (vm[j]<0)*vm[j] )/dz_ave  
            Ci[j] += +( (vm[j-1]>0)*vm[j-1] )/dz_ave  
            # advective component using upwind
 
        tmp0 = (A[0] + Ai[0])*y[0] + (B[0] + Bi[0])*y[1] # shape of ni-long 1D array  
        tmp1 = np.ndarray.flatten( (np.vstack(A[1:nz-1])*y[1:(nz-1)] + np.vstack(B[1:nz-1])*y[1+1:(nz-1)+1] + np.vstack(C[1:nz-1])*y[1-1:(nz-1)-1]) ) 
        tmp1 += np.ndarray.flatten( Ai[1:nz-1]*y[1:(nz-1)] + Bi[1:nz-1]*y[1+1:(nz-1)+1] + Ci[1:nz-1]*y[1-1:(nz-1)-1] ) # shape of (nz-2,ni)
        tmp2 = (A[nz-1] + Ai[nz-1])*y[nz-1] + (C[nz-1] + Ci[nz-1])*y[nz-2]
        diff = np.append(np.append(tmp0, tmp1), tmp2)
        diff = diff.reshape(nz,ni)

        if vulcan_cfg.use_topflux == True:
            # Don't forget dz!!! -d phi/ dz
            ### the const flux has no contribution to the jacobian ### 
            diff[-1] += atm.top_flux /dzi[-1]
        if vulcan_cfg.use_botflux == True:
            ### the deposition term needs to be included in the jacobian!!!   
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) /dzi[0]
        
        return diff

    def diffdf_settling(self, y, atm): 
        """
        function of eddy diffusion including molecular diffusion and the settling velocity for particles, with zero-flux boundary conditions and non-uniform grids (dzi)
        in the form of Aj*y_j + Bj+1*y_j+1 + Cj-1*y_j-1
        """
        
        y = y.copy()
        
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
    
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        Dzz = atm.Dzz.copy()
        vs = atm.vs.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        ms = atm.ms.copy()
        Hp = atm.Hp.copy()
        g = atm.g
        Ti = atm.Ti
        Hpi = atm.Hpi
        
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)
        Ai, Bi, Ci = [ np.zeros((nz,ni)) for i in range(3)]
        
        A[0] = -1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]     
        B[0] = 1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] 
        C[0] = 0 
        A[nz-1] = -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] 
        B[nz-1] = 0 
        C[nz-1] = 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] 
        
        # vertical adection (with closed B.C.) 
        A[0] += -( (vz[0]>0)*vz[0] )/dzi[0]
        B[0] += -( (vz[0]<0)*vz[0] )/dzi[0]
        A[-1] += ( (vz[-1]<0)*vz[-1] )/dzi[-1]
        C[-1] += ( (vz[-1]>0)*vz[-1] )/dzi[-1]
        # vertical adection
        
        # shape of ni-long 1D array
        # Including the settling velocity of the particles
        Ai[0] = -1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0] +\
        1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )  -( (vs[0]>0)*vs[0] )/dzi[0]  
        Bi[0] = 1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] +\
        1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )  -( (vs[0]<0)*vs[0] )/dzi[0]
        #Ci[0] = 0 
        Ai[nz-1] = -1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] \
        -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )  +( (vs[-1]<0)*vs[-1] )/dzi[-1]
        #Bi[nz-1] = 0
        Ci[nz-1] = 1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] \
        -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )  +( (vs[-1]>0)*vs[-1] )/dzi[-1]
        
        for j in range(1,nz-1):
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            A[j] = -1./dz_ave * ( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            B[j] = 1./dz_ave * Kzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            C[j] = 1./dz_ave * Kzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # vertical adection
            A[j] += -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            B[j] += -( (vz[j]<0)*vz[j] )/dz_ave
            C[j] += ( (vz[j-1]>0)*vz[j-1] )/dz_ave
            # vertical adection
            
            # Ai in the shape of nz*ni and Ai[j] in the shape of ni 
            # Including the settling velocity of the particles
            Ai[j] = -1./dz_ave * ( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j] -( (vs[j]>0)*vs[j] - (vs[j-1]<0)*vs[j-1] )/dz_ave
            Bi[j] = 1./dz_ave * Dzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]  -( (vs[j]<0)*vs[j] )/dz_ave
            Ci[j] = 1./dz_ave * Dzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]  +( (vs[j-1]>0)*vs[j-1] )/dz_ave
            
            Ai[j] += 1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g[j]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
            - Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j]/(Navo*kb*Ti[j-1])+ alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) ) #/ysum[j]
            Bi[j] += 1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g[j+1]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )
            Ci[j] += -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j-1]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )
 
        tmp0 = (A[0] + Ai[0])*y[0] + (B[0] + Bi[0])*y[1] # shape of ni-long 1D array  
        tmp1 = np.ndarray.flatten( (np.vstack(A[1:nz-1])*y[1:(nz-1)] + np.vstack(B[1:nz-1])*y[1+1:(nz-1)+1] + np.vstack(C[1:nz-1])*y[1-1:(nz-1)-1]) ) 
        tmp1 += np.ndarray.flatten( Ai[1:nz-1]*y[1:(nz-1)] + Bi[1:nz-1]*y[1+1:(nz-1)+1] + Ci[1:nz-1]*y[1-1:(nz-1)-1] ) # shape of (nz-2,ni)
        tmp2 = (A[nz-1] + Ai[nz-1])*y[nz-1] + (C[nz-1] + Ci[nz-1])*y[nz-2]
        diff = np.append(np.append(tmp0, tmp1), tmp2)
        diff = diff.reshape(nz,ni)

        if vulcan_cfg.use_topflux == True:
            # Don't forget dz!!! -d phi/ dz
            ### the const flux has no contribution to the jacobian ### 
            diff[-1] += atm.top_flux /dzi[-1]
        if vulcan_cfg.use_botflux == True:
            ### the deposition term needs to be included in the jacobian!!!   
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) /dzi[0]
        
        return diff
            
    
    def diffdf_settling_vm(self, y, atm): 
        """
        added vm for molecular diffusion
        function of eddy diffusion including molecular diffusion and the settling velocity for particles, with zero-flux boundary conditions and non-uniform grids (dzi)
        in the form of Aj*y_j + Bj+1*y_j+1 + Cj-1*y_j-1
        """
        
        y = y.copy()
        
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        Dzz = atm.Dzz.copy()
        vs = atm.vs.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        ms = atm.ms.copy()
        Hp = atm.Hp.copy()
        g = atm.g
        Ti = atm.Ti
        Hpi = atm.Hpi
        
        vm = atm.vm
        # shape: nz x ni
        # vm defined in build.py
        # vm = - Dzz_cen * ( ms[np.newaxis,:]*g[:,np.newaxis]/(Navo*kb*Tco[:,np.newaxis]) - 1./Hp[:,np.newaxis] +  alpha/Tco[:,np.newaxis]*(delta_T[:,np.newaxis])/dz[:,np.newaxis]  )

            
        A, B, C = np.zeros(nz), np.zeros(nz), np.zeros(nz)
        Ai, Bi, Ci = [ np.zeros((nz,ni)) for i in range(3)]
        
        A[0] = -1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0]     
        B[0] = 1./(dzi[0])*(Kzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] 
        C[0] = 0 
        A[nz-1] = -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] 
        B[nz-1] = 0 
        C[nz-1] = 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] 
        
        # vertical adection (with closed B.C.) 
        A[0] += -( (vz[0]>0)*vz[0] )/dzi[0]
        B[0] += -( (vz[0]<0)*vz[0] )/dzi[0]
        A[-1] += ( (vz[-1]<0)*vz[-1] )/dzi[-1]
        C[-1] += ( (vz[-1]>0)*vz[-1] )/dzi[-1]
        # vertical adection
        
        # shape of ni-long 1D array
        # Including the settling velocity of the particles and the advective component of molecular diffusion
        Ai[0] = -1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[0] \
        -( (vs[0]>0)*vs[0] )/dzi[0]  
        Bi[0] = 1./(dzi[0])*(Dzz[0]/dzi[0]) *(ysum[1]+ysum[0])/2. /ysum[1] \
        -( (vs[0]<0)*vs[0] )/dzi[0]
        #Ci[0] = 0 
        Ai[nz-1] = -1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-1] \
        +( (vm[-1]<0)*vm[-1] )/dzi[-1]  +( (vs[-1]<0)*vs[-1] )/dzi[-1]
        #Bi[nz-1] = 0
        Ci[nz-1] = 1./(dzi[-1])*(Dzz[nz-2]/dzi[-1]) *(ysum[nz-1]+ysum[nz-2])/2. /ysum[nz-2] \
        +( (vm[-1]>0)*vm[-1] )/dzi[-1]  +( (vs[-1]>0)*vs[-1] )/dzi[-1]
        
        for j in range(1,nz-1):
            dz_ave = 0.5*(dzi[j-1] + dzi[j])
            A[j] = -1./dz_ave * ( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j]  
            B[j] = 1./dz_ave * Kzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]
            C[j] = 1./dz_ave * Kzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]
            
            # vertical adection
            A[j] += -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
            B[j] += -( (vz[j]<0)*vz[j] )/dz_ave
            C[j] += ( (vz[j-1]>0)*vz[j-1] )/dz_ave
            # vertical adection
            
            # Ai in the shape of nz*ni and Ai[j] in the shape of ni 
            # Including the settling velocity of the particles
            
            # diffusion component
            Ai[j] = -1./dz_ave * ( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j]+ysum[j-1])/2. ) /ysum[j] 
            Bi[j] = 1./dz_ave * Dzz[j]/dzi[j] *(ysum[j+1]+ysum[j])/2. /ysum[j+1]  
            Ci[j] = 1./dz_ave * Dzz[j-1]/dzi[j-1] *(ysum[j]+ysum[j-1])/2. /ysum[j-1]  
            # diffusion component
            
            # advective component using upwind (inc. from Dzz and from vs)
            Ai[j] += -( (vm[j]>0)*vm[j] - (vm[j-1]<0)*vm[j-1] )/dz_ave  -( (vs[j]>0)*vs[j] - (vs[j-1]<0)*vs[j-1] )/dz_ave
            Bi[j] += -( (vm[j]<0)*vm[j] )/dz_ave  -( (vs[j]<0)*vs[j] )/dz_ave
            Ci[j] += +( (vm[j-1]>0)*vm[j-1] )/dz_ave  +( (vs[j-1]>0)*vs[j-1] )/dz_ave 
            # advective component using upwind
            
        tmp0 = (A[0] + Ai[0])*y[0] + (B[0] + Bi[0])*y[1] # shape of ni-long 1D array  
        tmp1 = np.ndarray.flatten( (np.vstack(A[1:nz-1])*y[1:(nz-1)] + np.vstack(B[1:nz-1])*y[1+1:(nz-1)+1] + np.vstack(C[1:nz-1])*y[1-1:(nz-1)-1]) ) 
        tmp1 += np.ndarray.flatten( Ai[1:nz-1]*y[1:(nz-1)] + Bi[1:nz-1]*y[1+1:(nz-1)+1] + Ci[1:nz-1]*y[1-1:(nz-1)-1] ) # shape of (nz-2,ni)
        tmp2 = (A[nz-1] + Ai[nz-1])*y[nz-1] + (C[nz-1] + Ci[nz-1])*y[nz-2]
        diff = np.append(np.append(tmp0, tmp1), tmp2)
        diff = diff.reshape(nz,ni)

        if vulcan_cfg.use_topflux == True:
            # Don't forget dz!!! -d phi/ dz
            ### the const flux has no contribution to the jacobian ### 
            diff[-1] += atm.top_flux /dzi[-1]
        if vulcan_cfg.use_botflux == True:
            ### the deposition term needs to be included in the jacobian!!!   
            diff[0] += (atm.bot_flux - y[0]*atm.bot_vdep) /dzi[0]
        
        return diff
        
        
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

    def lhs_jac_tot_vm(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion)
        zero-flux BC:  1st derivitive of y is zero
        inc. vm from molecular diffusion
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
        vm = atm.vm
        
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
            -( (vm[j]>0)*vm[j] - (vm[j-1]<0)*vm[j-1] )/dz_ave
            dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
            -( (vm[j]<0)*vm[j] )/dz_ave
            dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
            +( (vm[j-1]>0)*vm[j-1] )/dz_ave
    
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vm[0]>0)*vm[0] )/dzi[0]
        # deposition velocity
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        # diffusion-limited escape
        if vulcan_cfg.diff_esc: # not empty list
            diff_lim = np.zeros(ni)
            for sp in vulcan_cfg.diff_esc:
                if y[-1,species.index(sp)] > 0:
                    diff_lim[species.index(sp)] += atm.top_flux[species.index(sp)] /y[-1,species.index(sp)]
            dfdy[j_indx[-1], j_indx[-1]] -= diff_lim # negative
            
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vm[0]<0)*vm[0] )/dzi[0]

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
        +( (vm[-1]<0)*vm[-1] )/dzi[-1]
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
        +( (vm[-1]>0)*vm[-1] )/dzi[-1]

        return dfdy

        
    def lhs_jac_no_mol(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy 
        jacobian matrix for dn/dt + dphi/dz = P - L (WITHOUT molecular diffusion)
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
        vz = atm.vz.copy()
        Tco = atm.Tco.copy()
        mu, ms = atm.mu.copy(),  atm.ms.copy()

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
    
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        # deposition velocity
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1] 
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]  

        return dfdy
    
    def lhs_jac_fix_all_bot(self, var, atm):
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion)
        Fixed all species BC: all species at bottom (y[0]) remains fixed
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

        Ti = atm.Ti.copy()
        Hpi = atm.Hpi.copy()

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
    
        # deposition velocity (off with fixed all BC)
        # if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        # diffusion-limited escape
        if vulcan_cfg.diff_esc: # not empty list
            diff_lim = np.zeros(ni)
            for sp in vulcan_cfg.diff_esc:
                if y[-1,species.index(sp)] > 0:
                    diff_lim[species.index(sp)] += atm.top_flux[species.index(sp)] /y[-1,species.index(sp)]
            dfdy[j_indx[-1], j_indx[-1]] -= diff_lim # negative
            
        # Fix bottom BC
        #print (dfdy[:, j_indx[0]])
        dfdy[:, j_indx[0]] = 0.
        
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
        
    def lhs_jac_no_mol_fix_all_bot(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy 
        jacobian matrix for dn/dt + dphi/dz = P - L (WITHOUT molecular diffusion)
        Fixed all species BC: all species at bottom (y[0]) remains fixed
        """
        y = var.y.copy()
        # TEST condensation excluding non-gaseous species
        if vulcan_cfg.non_gas_sp:
            ysum = np.sum(y[:,atm.gas_indx], axis=1)
        else: ysum = np.sum(y, axis=1)
        # TEST condensation excluding non-gaseous species
        dzi = atm.dzi.copy()
        Kzz = atm.Kzz.copy()
        vz = atm.vz.copy()
        Tco = atm.Tco.copy()
        mu, ms = atm.mu.copy(),  atm.ms.copy()

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
    
        #dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        # deposition velocity (off with fixed all BC)
        # if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        # Fix bottom BC
        dfdy[:, j_indx[0]] = 0.
        
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1] 
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]  

        return dfdy

    def lhs_jac_settling(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion and gravitation settling for particles)
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
        vs = atm.vs.copy()
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
            - Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) )  -( (vs[j]>0)*vs[j] - (vs[j-1]<0)*vs[j-1] )/dz_ave
            dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
            +1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g[j+1]/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )  -( (vs[j]<0)*vs[j] )/dz_ave
            dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
            -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g[j-1]/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )  +( (vs[j-1]>0)*vs[j-1] )/dz_ave
    
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )  -( (vs[0]>0)*vs[0] )/dzi[0]
        # deposition velocity
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0] 
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
        +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g[0]/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] ) -( (vs[0]<0)*vs[0] )/dzi[0]

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
        - 1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] ) +( (vs[-1]<0)*vs[-1] )/dzi[-1]
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]   
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
                -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g[-1]/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] ) +( (vs[-1]>0)*vs[-1] )/dzi[-1]

        return dfdy
                
    def lhs_jac_settling_vm(self, var, atm):      
        """
        directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
        jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion and gravitation settling for particles)
        zero-flux BC:  1st derivitive of y is zero
        inc. vs from molecular diffusion
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
        vs = atm.vs.copy()
        alpha = atm.alpha.copy()
        Tco = atm.Tco.copy()
        mu, ms = atm.mu.copy(),  atm.ms.copy()
        g = atm.g
        vm = atm.vm

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
            -( (vs[j]>0)*vs[j] - (vs[j-1]<0)*vs[j-1] )/dz_ave  -( (vm[j]>0)*vm[j] - (vm[j-1]<0)*vm[j-1] )/dz_ave
            dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
            -( (vs[j]<0)*vs[j] )/dz_ave -( (vm[j]<0)*vm[j] )/dz_ave
            dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
            +( (vs[j-1]>0)*vs[j-1] )/dz_ave +( (vm[j-1]>0)*vm[j-1] )/dz_ave
    
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
        dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) \
        -( (vs[0]>0)*vs[0] )/dzi[0]
        # deposition velocity
        if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
        
        # diffusion-limited escape
        if vulcan_cfg.diff_esc: # not empty list
            diff_lim = np.zeros(ni)
            for sp in vulcan_cfg.diff_esc:
                if y[-1,species.index(sp)] > 0:
                    diff_lim[species.index(sp)] += atm.top_flux[species.index(sp)] /y[-1,species.index(sp)]
            dfdy[j_indx[-1], j_indx[-1]] -= diff_lim # negative
            
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0] 
        dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
         -( (vs[0]<0)*vs[0] )/dzi[0]

        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]  
        dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
        +( (vs[-1]<0)*vs[-1] )/dzi[-1]  +( (vm[-1]<0)*vm[-1] )/dzi[-1]
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]   
        dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
        +( (vs[-1]>0)*vs[-1] )/dzi[-1]  +( (vm[-1]>0)*vm[-1] )/dzi[-1]

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
                diffdf = self.diffdf
                jac_tot = self.lhs_jac_tot
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf = self.diffdf_settling
                jac_tot = self.lhs_jac_settling
            else:
                diffdf = self.diffdf_no_mol
                jac_tot = self.lhs_jac_no_mol
        else: # vulcan_cfg.use_vm_mol == True:
            if vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == False:
                diffdf = self.diffdf_vm
                jac_tot = self.lhs_jac_tot_vm
            elif vulcan_cfg.use_moldiff == True and vulcan_cfg.use_settling == True:
                diffdf = self.diffdf_settling_vm
                jac_tot = self.lhs_jac_settling_vm
            else:
                diffdf = self.diffdf_no_mol
                jac_tot = self.lhs_jac_no_mol
            
        r = 1. + 1./2.**0.5

        df = chemdf(y,M,k).flatten() + diffdf(y, atm).flatten()
        lhs = jac_tot(var, atm)
        
        # Fixed species including only below the cold trap # TEST 2022
        if vulcan_cfg.use_condense == True and para.fix_species_start == True:
            for sp in vulcan_cfg.fix_species:
                if vulcan_cfg.fix_species_from_coldtrap_lev == False: # if Ptop is not specified, fix the whole column # TEST2022
                    pass
                else:
                    pfix_indx = atm.conden_min_lev[sp]
                    atm.fix_sp_indx[sp] = np.arange(species.index(sp), species.index(sp) + ni*(pfix_indx), ni)
                
                df[atm.fix_sp_indx[sp]] = 0
                lhs[atm.fix_sp_indx[sp],:] = 0
                lhs[atm.fix_sp_indx[sp],atm.fix_sp_indx[sp]] = 1./(r*h)  # cuz the jacobian func is directly outputing 1./(r*h)*sparse.identity(ni*nz) - dfdy                        
        
        if vulcan_cfg.use_ion == True:
            df[atm.fix_e_indx] = 0
            lhs[atm.fix_e_indx,:] = 0
            lhs[atm.fix_e_indx,atm.fix_e_indx] = 1./(r*h)
        
        lhs_b, bw = self.store_bandM(lhs,ni,nz)
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
            
    
