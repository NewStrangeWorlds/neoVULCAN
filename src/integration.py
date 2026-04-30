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

class Integration(object):
    """
    time-stepping until the stopping criteria (steady-state) is satisfied
    #all the operators required in the continuity equation: dn/dt + dphi/dz = P - L
    #or class incorporating the esential numerical operations?
    """
    
    def __init__(self, odesolver, output):

        self.mtol = vulcan_cfg.mtol
        self.atol = vulcan_cfg.atol
        self.output = output
         
        self.odesolver = odesolver
        self.non_gas_sp = vulcan_cfg.non_gas_sp
        self.use_settling = vulcan_cfg.use_settling 
        
        # including photoionisation
        if vulcan_cfg.use_photo == True: self.update_photo_frq = vulcan_cfg.ini_update_photo_frq
        
        if vulcan_cfg.use_condense == True:  
            self.non_gas_sp_index = [species.index(sp) for sp in self.non_gas_sp]
            self.condense_sp_index = [species.index(sp) for sp in vulcan_cfg.condense_sp]
        
        
    def __call__(self, var, atm, para, make_atm):
        
        use_print_prog, use_live_plot = vulcan_cfg.use_print_prog, vulcan_cfg.use_live_plot
        self.loss_criteria = 0.0005
        
        while not self.stop(var, para, atm): # Looping until the stop condition is satisfied
            
            var = self.backup(var)
            
            # updating tau, flux, and the photolosys rate
            # swtiching to final_update_photo_frq
            if vulcan_cfg.use_photo == True and var.longdy < vulcan_cfg.yconv_min*10. and var.longdydt < 1.e-6:  
                self.update_photo_frq = vulcan_cfg.final_update_photo_frq
                if para.switch_final_photo_frq == False:
                    print ('update_photo_frq changed to ' + str(vulcan_cfg.final_update_photo_frq) +'\n')
                    para.switch_final_photo_frq = True
            
            if vulcan_cfg.use_photo == True and para.count % self.update_photo_frq == 0:
                self.odesolver.compute_tau(var, atm)
                self.odesolver.compute_flux(var, atm)
                self.odesolver.compute_J(var, atm)
                if vulcan_cfg.use_ion == True: # photoionisation rate
                    self.odesolver.compute_Jion(var, atm)
                                    
            # integrating one step
            var, para = self.odesolver.one_step(var, atm, para)
            
            # # TEST 2025: using atom_loss to reduce rtol
            if vulcan_cfg.use_adapt_rtol == True and para.count%10 == 0:
                if max([np.abs(loss) for loss in var.atom_loss.values()]) >= self.loss_criteria: 
                    self.loss_criteria *= 2.
                    vulcan_cfg.rtol *= 0.75
                    vulcan_cfg.rtol = max(vulcan_cfg.rtol, vulcan_cfg.rtol_min)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_min:
                        print ('rtol reduced to ' + str(vulcan_cfg.rtol))
                        print ('------------------------------------------------------------------')

            if vulcan_cfg.use_adapt_rtol == True and para.count%1000 == 0 and para.count>0:
                if max([np.abs(loss) for loss in var.atom_loss.values()]) < 2e-4: #
                    vulcan_cfg.rtol *= 1.25
                    vulcan_cfg.rtol = min(vulcan_cfg.rtol, vulcan_cfg.rtol_max)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_max:
                        print ('rtol increased to ' + str(vulcan_cfg.rtol))
                        print ('------------------------------------------------------------------')
            #
            # # TEST 2025
            
            # Condensation (needs to be after solver.one_step)
            if vulcan_cfg.use_condense == True and var.t >= vulcan_cfg.start_conden_time and para.fix_species_start == False:
                # updating the condensation rates 
                var = self.conden(var,atm)
                
                if vulcan_cfg.fix_species and var.t > vulcan_cfg.stop_conden_time:
                    
                    if para.fix_species_start == False: # switch to run for the first time
                        
                        para.fix_species_start = True
                        vulcan_cfg.rtol = vulcan_cfg.post_conden_rtol
                        print ("rtol changed to " + str(vulcan_cfg.rtol) + " after fixing the condensaed species.")
                        atm.vs *= 0
                        print ("Turn off the settling velocity of all species")
                        # updated 2023
                        
                        var.fix_y = {}
                        for sp in vulcan_cfg.fix_species:
                            var.fix_y[sp] = np.copy(var.y[:,species.index(sp)]) 
                            
                            # record the cold trap levels
                            if vulcan_cfg.fix_species_from_coldtrap_lev == True:
                                
                                if sp == 'H2O_l_s' or sp == 'H2SO4_l' or sp == 'NH3_l_s' or sp == 'S8_l_s': atm.conden_min_lev[sp] = nz-1 # fix condensates through the whole amtosphere 
                                    # updated 2023
                                else:                                
                                    sat_rho = atm.n_0 * atm.sat_mix[sp]
                                    conden_status = var.y[:,species.index(sp)] >= sat_rho
                                    atm.conden_status = conden_status 
                                    
                                    if list(var.y[conden_status,species.index(sp)]): # if it condenses
                                        min_sat = np.amin(atm.sat_mix[sp][conden_status]) # the mininum value of the saturation p within the saturation region
                                        atm.min_sat = min_sat
                                        conden_min_lev = np.where(atm.sat_mix[sp] == min_sat)[0].item()
                                        atm.conden_min_lev[sp] = conden_min_lev
                                        print (sp + " is now fixed from " + "{:.2e}".format(atm.pco[atm.conden_min_lev[sp]]/1e6) + " bar." )
                                    else:
                                        print (sp + " not condensed.")
                                        atm.conden_min_lev[sp] = 0
                                    
                    else: pass # do nothing after fix_species has started
                
                # this is inside the fix_species_start == False loop        
                if vulcan_cfg.use_relax:
                    if 'H2O' in vulcan_cfg.use_relax: 
                        var = self.h2o_conden_evap_relax(var,atm)
                    if 'NH3' in vulcan_cfg.use_relax:
                        var = self.nh3_conden_evap_relax(var,atm)
                                          
            if para.count % vulcan_cfg.update_frq == 0: # updating mu and dz (dzi) due to diffusion
                atm = self.update_mu_dz(var, atm, make_atm)
                atm = self.update_phi_esc(var, atm) # updating the diffusion-limited flux
                
            # MAINTAINING HYDROSTATIC BALANCE
            if vulcan_cfg.use_condense == True:
                #var.v_ratio = np.sum(var.y[:,atm.gas_indx], axis=1) / atm.n_0
                var.y[:,atm.gas_indx] = np.vstack(atm.n_0)*var.ymix[:,atm.gas_indx]
            else:
                #var.v_ratio = np.sum(var.y, axis=1) / atm.n_0 # how the volumn has changed while the P and number density are fixed
                var.y = np.vstack(atm.n_0)*var.ymix
            
            # calculating the changing of y
            var = self.f_dy(var, para)
            
            # save values of the current step
            var, para = self.save_step(var, para)
            
            # adjusting the step-size
            var = self.odesolver.step_size(var, para)
            
            if use_print_prog == True and para.count % vulcan_cfg.print_prog_num==0:
                self.output.print_prog(var,para)
                
            if vulcan_cfg.use_live_flux == True and vulcan_cfg.use_photo == True and para.count % vulcan_cfg.live_plot_frq ==0:
                #plt.figure('flux')
                self.output.plot_flux_update(var, atm, para)
                
            if use_live_plot == True and para.count % vulcan_cfg.live_plot_frq ==0:
                #plt.figure('mix')
                self.output.plot_update(var, atm, para)
            
            
        
    def backup(self, var):
        var.y_prev = np.copy(var.y)
        var.dy_prev = np.copy(var.dy)
        var.atom_loss_prev = var.atom_loss.copy()
        return var
        
    def update_mu_dz(self, var, atm, make_atm): #y, ni, spec, Tco, pco
        
        # gravity
        gz = atm.g
        pref_indx = atm.pref_indx
        Tco, pico = atm.Tco.copy(), atm.pico.copy()
        # calculating mu (mean molecular weight)
        atm = make_atm.mean_mass(var, atm, ni)
        Hp = atm.Hp
        
        for i in range(pref_indx,nz):
            if i == pref_indx:
                atm.g[i] = atm.gs
                Hp[i] = kb*Tco[i]/(atm.mu[i]/Navo*atm.gs)    
            else:
                atm.g[i] = atm.gs * (vulcan_cfg.Rp/(vulcan_cfg.Rp+ atm.zco[i]))**2
                Hp[i] = kb*Tco[i]/(atm.mu[i]/Navo*atm.g[i])
            atm.dz[i] = Hp[i] * np.log(pico[i]/pico[i+1]) # pico[i+1] has a lower P than pico[i] (higer height)
            atm.zco[i+1] = atm.zco[i] + atm.dz[i] # zco is set zero at 1bar for gas giants

        # for pref_indx != zero 
        if not pref_indx == 0:
            for i in range(pref_indx-1,-1,-1):
                atm.g[i] = atm.gs * (vulcan_cfg.Rp/(vulcan_cfg.Rp + atm.zco[i+1]))**2
                Hp[i] = kb*Tco[i]/(atm.mu[i]/Navo*atm.g[i])
                atm.dz[i] = Hp[i] * np.log(pico[i]/pico[i+1]) 
                atm.zco[i] = atm.zco[i+1] - atm.dz[i] # from i+1 propogating down to i
            
        zmco = 0.5*(atm.zco + np.roll(atm.zco,-1))
        atm.zmco = zmco[:-1]
        dzi = 0.5*(atm.dz + np.roll(atm.dz,1))
        atm.dzi = dzi[1:]
        
        # for the molecular diffsuion
        if vulcan_cfg.use_moldiff == True:
            Ti = 0.5*(Tco + np.roll(Tco,-1))
            atm.Ti = Ti[:-1]
            Hpi = 0.5*(Hp + np.roll(Hp,-1))
            atm.Hpi = Hpi[:-1]
        
        return atm
    
    def update_phi_esc(self, var, atm): # updating diffusion-mimited escape
    
        # Diffusion limited escape 
        for sp in vulcan_cfg.diff_esc:

            #atm.top_flux[species.index(sp)] = - atm.Dzz[-1,species.index(sp)] *var.y[-1,species.index(sp)] /atm.Hp[-1]
            atm.top_flux[species.index(sp)] = - atm.Dzz[-1,species.index(sp)]*var.y[-1,species.index(sp)]*( 1./atm.Hp[-1] -atm.ms[species.index(sp)]* atm.g[-1]/(Navo*kb*atm.Tco[-1])     )            
            atm.top_flux[species.index(sp)] = max(atm.top_flux[species.index(sp)], vulcan_cfg.max_flux*(-1))
            
            # print ("Escape flux of " + sp + "{:>10.2e}".format(atm.top_flux[species.index(sp)]))
            # print ("diffusion-limite value: " + "{:>10.2e}".format(- atm.Dzz[-1,species.index(sp)]*var.y[-1,species.index(sp)]*( 1./atm.Hp[-1] -atm.ms[species.index(sp)]* atm.g[-1]/(Navo*kb*atm.Tco[-1])     )) )
            #print ("Test  " + sp + "{:>10.2e}".format(atm.Dzz[-1,species.index(sp)] *var.y[-1,species.index(sp)] /atm.Hp[-1]) )
            
        return atm
    
    
    # function calculating the change of y
    def f_dy(self, var, para): # y, y_prev, ymix, yconv, count, dt
        if para.count == 0: 
            var.dy, var.dydt = 1., 1.
            return var
        y, ymix, y_prev = var.y, var.ymix, var.y_prev    
        dy =  np.abs(y - y_prev)
        dy[ymix < vulcan_cfg.mtol] = 0   
        dy[y < vulcan_cfg.atol] = 0 
        dy = np.amax( dy[y>0]/y[y>0] )
        
        var.dy, var.dydt = dy, dy/var.dt
        
        return var
    
    
    def conv(self, var, para, atm, out=False, print_freq=100):
        '''
        funtion returns TRUE if the convergence condition is satisfied
        '''
        st_factor, mtol_conv, atol, yconv_cri, slope_cri, yconv_min =\
         vulcan_cfg.st_factor, vulcan_cfg.mtol_conv, vulcan_cfg.atol, vulcan_cfg.yconv_cri, vulcan_cfg.slope_cri, vulcan_cfg.yconv_min
        y, ymix, y_time, t_time = var.y.copy(), var.ymix.copy(), var.y_time, var.t_time
        count = para.count
        
        #slope_min = min( np.amin(atm.Kzz)/np.amax(0.1*atm.Hp)**2 , 1.e-8)
        slope_min = min( np.amin(atm.Kzz/(0.1*atm.Hp[:-1])**2) , 1.e-8)
        slope_min = max(slope_min, 1.e-10)

        indx = np.abs(t_time-var.t*st_factor).argmin()    
        if indx == para.count-1: indx-=1  #Important!! For dt larger than half of the runtime (count-1 is the last one) 
        
        # Don't check more than vulcan_cfg.conv_step (1000) steps back 
        indx = max(para.count-vulcan_cfg.conv_step, indx)
        
        # TEST
        if para.count %100==0: print ("conv_indx: "  + str(indx))
        
        longdy = np.abs((y_time[count-1] - y_time[indx])/np.vstack(atm.n_0))
        longdy[ymix < mtol_conv] = 0
        longdy[y < atol] = 0
        
        # to get rid off non-convergent species, e.g. HC3N without sinks
        if 'conver_ignore' in dir(vulcan_cfg):
            for sp in vulcan_cfg.conver_ignore: longdy[:,species.index(sp)] = 0 # added 2023 
        
        if vulcan_cfg.use_condense == True:
            longdy[:,self.non_gas_sp_index] = 0
        
        with np.errstate(divide='ignore',invalid='ignore'): # ignoring nan when devided by zero
            where_varies_most = longdy/ymix
        para.where_varies_most = where_varies_most
         
        longdy = np.amax( longdy[ymix>0]/ymix[ymix>0] )
        longdydt = longdy/(t_time[-1]-t_time[indx])
        # store longdy and longdydt
        var.longdy, var.longdydt = longdy, longdydt
        
        if (longdy < yconv_cri and longdydt < slope_cri or longdy < yconv_min and longdydt < slope_min) and var.aflux_change<vulcan_cfg.flux_cri: 
            return True
            
        return False
    
    def stop(self, var, para, atm):
        '''
        To check the convergence criteria and stop the integration 
        '''
        if var.t > vulcan_cfg.trun_min and para.count > vulcan_cfg.count_min and self.conv(var, para, atm):
            print ('Integration successful with ' + str(para.count) + ' steps and long dy, long dydt = ' + str(var.longdy) + ' ,' + str(var.longdydt) + '\nActinic flux change: ' + '{:.2E}'.format(var.aflux_change)) 
            self.output.print_end_msg(var, para)
            para.end_case = 1
            return True
        elif var.t > vulcan_cfg.runtime:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print ('Integration not completed...\nMaximal allowed runtime exceeded ('+ \
            str (vulcan_cfg.runtime) + ' sec)!')
            para.end_case = 2
            return True
        elif para.count > vulcan_cfg.count_max:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print ('Integration not completed...\nMaximal allowed steps exceeded (' + \
            str (vulcan_cfg.count_max) + ')!')
            para.end_case = 3
            return True
    
    def save_step(self, var, para):
        '''
        save current values of y and add 1 to the counter
        '''
        var.t += var.dt   
        para.count += 1
        
        # tmp = list(var.y)
        #if para.count % self.y_time_freq ==0:
        var.y_time.append(var.y)
        #var.ymix_time.append(var.ymix.copy())
        var.t_time.append(var.t)
    
        # only used in PI_control
        # var.dy_time.append(var.y)
        # var.dydt_time.append(var.dydt)
        var.atom_loss_time.append(list(var.atom_loss.values()) )
        
        return var, para
        
    
    # TESTing condensation
    def conden(self, var, atm):
        '''
        Updating the condensation reactions according to the new number density
        using the condensation growth timescale in the contiuum regime (not in the kinetic regime)
        
        Note that when n_g -> n_s, n_s is still the number density of "molecules", not "particles."
        So I scaled down the evaporation rate by n_mol. 
        n_s / n_mol should also be used for plotting.
        '''
        for re in var.conden_re_list:
            if var.Rf[re] == 'H2O -> H2O_l_s' and 'H2O' in vulcan_cfg.condense_sp:
                # using realxation for water condensation
                if vulcan_cfg.use_relax:
                    var.k[re] = np.repeat(0.,nz)
                    var.k[re+1] = np.repeat(0.,nz)
                else:
                    m = 18./Navo
                    rho_p = atm.rho_p['H2O_l_s']
                    r_p = atm.r_p['H2O_l_s']
                    # relative humidity
                    sat_humidity = atm.sat_p['H2O']/kb/atm.Tco * vulcan_cfg.humidity

                    # this is based on the kinetic regime
                    rate_c = m/(4*rho_p)*(8*kb*atm.Tco/np.pi/m)**0.5 *(var.y[:,species.index('H2O')]-sat_humidity)/r_p

                    # new approach: contiuum regime DM/rho c
                    Dg = np.insert(atm.Dzz[:,species.index('H2O')], 0, atm.Dzz[0,species.index('H2O')])
                    rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('H2O')]-sat_humidity)

                    # how many gas molecules are contained in one particle with the assumed size r_p
                    n_mol = 4./3*np.pi*r_p**3 *rho_p /m

                    var.k[re] = rate
                    var.k[re+1] = rate #/n_mol

                    # positive: condensation
                    var.k[re] = np.maximum(var.k[re], 0)
                    # negative: evaporation
                    var.k[re+1] = np.minimum(var.k[re+1], 0)
                    var.k[re+1] = np.abs(var.k[re+1])
          
        
            elif var.Rf[re] == 'NH3 -> NH3_l' and 'NH3' in vulcan_cfg.condense_sp:
                # using realxation for water condensation
                if vulcan_cfg.use_relax:
                    var.k[re] = np.repeat(0.,nz)
                    var.k[re+1] = np.repeat(0.,nz)
                else:
                    m = 17./Navo
                    rho_p = atm.rho_p['NH3_l_s']
                    r_p = atm.r_p['NH3_l_s'] # assuming 1 micron
            
                    #rate_c = m/(4*rho_p)*(8*kb*atm.Tco/np.pi/m)**0.5 *(var.y[:,species.index('NH3')]-atm.sat_p['NH3']/kb/atm.Tco)/r_p
                
                    # new approach: contiuum regime DM/rho c
                    Dg = np.insert(atm.Dzz[:,species.index('NH3')], 0, atm.Dzz[0,species.index('NH3')])
                    rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('NH3')] - atm.sat_p['NH3']/kb/atm.Tco)
                
                    # how many gas molecules are contained in one particle with the assumed size r_p
                    n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                    var.k[re] = rate 
                    var.k[re+1] = rate #/n_mol

                    # positive: condensation
                    var.k[re] = np.maximum(var.k[re], 0)
                    # negative: evaporation
                    var.k[re+1] = np.minimum(var.k[re+1], 0)
                    var.k[re+1] = np.abs(var.k[re+1])
            
            elif var.Rf[re] == 'H2SO4 -> H2SO4_l' and 'H2SO4' in vulcan_cfg.condense_sp:
                m = 98.022/Navo
                rho_p = atm.rho_p['H2SO4_l']
                r_p = atm.r_p['H2SO4_l']
                
                # new approach: contiuum regime DM/rho c
                Dg = np.insert(atm.Dzz[:,species.index('H2SO4')], 0, atm.Dzz[0,species.index('H2SO4')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('H2SO4')] - atm.sat_p['H2SO4']/kb/atm.Tco)
                
                #rate_c = m/(4*rho_p)*(8*kb*atm.Tco/np.pi/m)**0.5 *(var.y[:,species.index('H2SO4')]-atm.sat_p['H2SO4']/kb/atm.Tco)/r_p
                
                # how many gas molecules are contained in one particle with the assumed size r_p
                n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                var.k[re] = rate 
                var.k[re+1] = rate #/n_mol

                # positive: condensation
                var.k[re] = np.maximum(var.k[re], 0)
                # negative: evaporation
                var.k[re+1] = np.minimum(var.k[re+1], 0)
                var.k[re+1] = np.abs(var.k[re+1])
                
            elif var.Rf[re] == 'S2 -> S2_l_s' and 'S2' in vulcan_cfg.condense_sp:
                m = 45.019/Navo 
                rho_p = atm.rho_p['S2_l_s']
                r_p = atm.r_p['S2_l_s']
                
                # new approach: contiuum regime DM/rho c
                Dg = np.insert(atm.Dzz[:,species.index('S2')], 0, atm.Dzz[0,species.index('S2')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('S2')] - atm.sat_p['S2']/kb/atm.Tco)
                
                # how many gas molecules are contained in one particle with the assumed size r_p
                n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                var.k[re] = rate 
                var.k[re+1] = rate #/n_mol

                # positive: condensation
                var.k[re] = np.maximum(var.k[re], 0)
                # negative: evaporation
                var.k[re+1] = np.minimum(var.k[re+1], 0)
                var.k[re+1] = np.abs(var.k[re+1])
            
            elif var.Rf[re] == 'S4 -> S4_l_s' and 'S4' in vulcan_cfg.condense_sp:
                m = 32.06*4/Navo 
                rho_p = atm.rho_p['S4_l_s']
                r_p = atm.r_p['S4_l_s']
                
                # new approach: contiuum regime DM/rho c
                Dg = np.insert(atm.Dzz[:,species.index('S4')], 0, atm.Dzz[0,species.index('S4')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('S4')] - atm.sat_p['S4']/kb/atm.Tco)
                
                # how many gas molecules are contained in one particle with the assumed size r_p
                n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                # acc_ratio = var.y[:,species.index('S4_l_s')]/n_mol /vulcan_cfg.n_ccn # accomdation ratio: 0 all ccn available 1=no more free ccn
                # lim_factor = 1-acc_ratio
                # lim_factor[lim_factor<0] = 0
                
                var.k[re] = rate #*lim_factor 
                var.k[re+1] = rate #/n_mol

                # positive: condensation
                var.k[re] = np.maximum(var.k[re], 0)
                # negative: evaporation
                var.k[re+1] = np.minimum(var.k[re+1], 0)
                var.k[re+1] = np.abs(var.k[re+1])
                    
            elif var.Rf[re] == 'S8 -> S8_l_s' and 'S8' in vulcan_cfg.condense_sp:
                m = 360.152/Navo
                rho_p = atm.rho_p['S8_l_s']
                r_p = atm.r_p['S8_l_s']
                
                # new approach: contiuum regime DM/rho c
                Dg = np.insert(atm.Dzz[:,species.index('S8')], 0, atm.Dzz[0,species.index('S8')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('S8')] - atm.sat_p['S8']/kb/atm.Tco)
                
                # how many gas molecules are contained in one particle with the assumed size r_p
                n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                var.k[re] = rate 
                var.k[re+1] = rate #/n_mol

                # positive: condensation
                var.k[re] = np.maximum(var.k[re], 0)
                # negative: evaporation
                var.k[re+1] = np.minimum(var.k[re+1], 0)
                var.k[re+1] = np.abs(var.k[re+1])
                
            elif var.Rf[re] == 'C -> C_s' and 'C' in vulcan_cfg.condense_sp:
                m = 12.011/Navo
                rho_p = atm.rho_p['C_s']
                r_p = atm.r_p['C_s']
                
                # new approach: contiuum regime DM/rho c
                Dg = np.insert(atm.Dzz[:,species.index('C')], 0, atm.Dzz[0,species.index('C')])
                rate = Dg * m/rho_p /r_p**2 * (var.y[:,species.index('C')] - atm.sat_p['C']/kb/atm.Tco)
                
                # how many gas molecules are contained in one particle with the assumed size r_p
                n_mol = 4./3*np.pi*r_p**3 *rho_p /m
                
                var.k[re] = rate 
                var.k[re+1] = rate #/n_mol

                # positive: condensation
                var.k[re] = np.maximum(var.k[re], 0)
                # negative: evaporation
                var.k[re+1] = np.minimum(var.k[re+1], 0)
                var.k[re+1] = np.abs(var.k[re+1])
                
            
        # for sp in vulcan_cfg.condense_sp:
#             atm.sat_mix[sp] = atm.sat_p[sp]/atm.pco
#             pre_conden = np.copy(var.y[:,species.index(sp)])
#             var.y[:,species.index(sp)] = np.minimum(atm.n_0 * atm.sat_mix[sp], var.y[:,species.index(sp)])
#             # storing the removed species
#             var.y_conden[:,species.index(sp)] += np.abs(pre_conden - var.y[:,species.index(sp)])
    
        return var
    
    def h2o_conden_relax(self, var, atm):
        m = 18./Navo
        rho_p = 0.95 # mix of water and ice
        r_p = atm.r_p['H2O_l_s'] 
        # relative humidity
        sat_humidity = atm.sat_p['H2O']/kb/atm.Tco * vulcan_cfg.humidity  
        
        # new approach: contiuum regime DM/rho c
        Dg = np.insert(atm.Dzz[:,species.index('H2O')], 0, atm.Dzz[0,species.index('H2O')])        
        tau = np.abs( 1./(Dg * m/rho_p /r_p**2 * (var.y[:,species.index('H2O')]-sat_humidity) ) )
        sat_mix = sat_humidity/atm.n_0

        # implicit-Euler to remove water
        y_conden = (var.ymix[:,species.index('H2O')] + var.dt/tau*sat_mix) / (1. + var.dt/tau)
        conden_indx = np.where( var.ymix[:,species.index('H2O')] > sat_mix )
        
        # how many gas molecules are contained in one particle with the assumed size r_p
        n_mol = 4./3*np.pi*r_p**3 *rho_p /m
        # and converting the mixing ratio of molecules /cm3 to droplets/cm3
        # "move" the condensed water to H2O_l_s
        var.ymix[conden_indx,species.index('H2O_l_s')] += (var.ymix[conden_indx,species.index('H2O')] - y_conden[conden_indx]) #/n_mol 
        
        var.ymix[conden_indx,species.index('H2O')] = y_conden[conden_indx]
        # restore the unsaturated parts (only relax where ymix > ysat)

        var.y = var.ymix * np.vstack( np.sum(var.y[:,atm.gas_indx], axis=1) )
        
        #print ("relax conden...")
        
        return var
    
    def h2o_conden_evap_relax(self, var, atm):
        m = 18./Navo
        rho_p = atm.rho_p['H2O_l_s'] # mix of water and ice
        r_p = atm.r_p['H2O_l_s'] 
        # relative humidity
        sat_humidity = atm.sat_p['H2O']/kb/atm.Tco * vulcan_cfg.humidity  
        
        # new approach: contiuum regime DM/rho c
        Dg = np.insert(atm.Dzz[:,species.index('H2O')], 0, atm.Dzz[0,species.index('H2O')])        
        tau = 1./(Dg * m/rho_p /r_p**2 * (var.y[:,species.index('H2O')]-sat_humidity) ) 
        conden_indx = np.where( tau > 0 )
        evap_indx = np.where(tau < 0)
        sat_mix = sat_humidity/atm.n_0
        #tau = np.abs(tau)
        
        # implicit-Euler to remove water
        y_conden = (var.ymix[:,species.index('H2O')] + var.dt/tau*sat_mix) / (1. + var.dt/tau)
        
        # evaporation to remove ice/water(liquid)
        ice_loss = (var.y[:,species.index('H2O')] - sat_humidity)*var.dt/tau # both tau < 0 and y_H2O - sat < 0 
        # cannot lose more than it has
        ice_loss = np.minimum(var.y[:,species.index('H2O_l_s')], ice_loss)
        
        # how many gas molecules are contained in one particle with the assumed size r_p
        # n_mol = 4./3*np.pi*r_p**3 *rho_p /m
        # and converting the mixing ratio of molecules /cm3 to droplets/cm3
        # "move" the condensed water to H2O_l_s
        var.ymix[conden_indx,species.index('H2O_l_s')] += (var.ymix[conden_indx,species.index('H2O')] - y_conden[conden_indx]) 
        var.ymix[conden_indx,species.index('H2O')] = y_conden[conden_indx]
        # store the saturated parts (only relax where ymix > ysat)
         
        var.ymix[evap_indx,species.index('H2O')] += ice_loss[evap_indx]/atm.n_0[evap_indx]
        var.ymix[evap_indx,species.index('H2O_l_s')] -= ice_loss[evap_indx]/atm.n_0[evap_indx]
        
        var.y = var.ymix * np.vstack( np.sum(var.y[:,atm.gas_indx], axis=1) )
        
        return var
    
    def nh3_conden_evap_relax(self, var, atm):
        m = 17./Navo
        rho_p = atm.rho_p['NH3_l_s'] # mix of water and ice
        r_p = atm.r_p['NH3_l_s'] 
        # relative humidity
        sat_p = atm.sat_p['NH3']/kb/atm.Tco 
        sat_mix = sat_p/atm.n_0
        
        conden_top = np.argmin(sat_mix)
        
        # new approach: contiuum regime DM/rho c
        Dg = np.insert(atm.Dzz[:,species.index('NH3')], 0, atm.Dzz[0,species.index('NH3')])        
        tau = 1./(Dg * m/rho_p /r_p**2 * (var.y[:,species.index('NH3')]-sat_p) ) 
        conden_indx = np.where( tau > 0 )[0]
        evap_indx = np.where(tau < 0)[0]
        
        # above the top of condensation zone, there should NOT be any condensation when using the relaxiation method
        conden_indx = [i for i in conden_indx if i <= conden_top]
        #evap_indx = [i for i in evap_indx if i <= conden_top]
        
        # implicit-Euler to remove water
        y_conden = (var.ymix[:,species.index('NH3')] + var.dt/tau*sat_mix) / (1. + var.dt/tau)
        
        # evaporation to remove ice/water(liquid)
        ice_loss =  (var.y[:,species.index('NH3')] - sat_p)*var.dt/tau # both tau < 0 and y_H2O - sat < 0 
        # cannot lose more than it has
        ice_loss = np.minimum(var.y[:,species.index('NH3_l_s')], ice_loss)
        
        # how many gas molecules are contained in one particle with the assumed size r_p
        # n_mol = 4./3*np.pi*r_p**3 *rho_p /m
        # and converting the mixing ratio of molecules /cm3 to droplets/cm3
        # "move" the condensed water to H2O_l_s
        var.ymix[conden_indx,species.index('NH3_l_s')] += (var.ymix[conden_indx,species.index('NH3')] - y_conden[conden_indx]) 
        var.ymix[conden_indx,species.index('NH3')] = y_conden[conden_indx]
        # store the saturated parts (only relax where ymix > ysat)
        
        
        # print ("Condex Indx:")
        # print (conden_indx)
        # print ("evap index:")
        # print (evap_indx)
        # print (ice_loss[evap_indx]/atm.n_0[evap_indx] /var.ymix[evap_indx,species.index('NH3_l_s')])
         
        var.ymix[evap_indx,species.index('NH3')] += ice_loss[evap_indx]/atm.n_0[evap_indx]
        # instaneous evaporation
        var.ymix[evap_indx,species.index('NH3_l_s')] -= ice_loss[evap_indx]/atm.n_0[evap_indx]
        #var.ymix[evap_indx,species.index('NH3_l_s')] = 0
        
        var.ymix[:,species.index('NH3_l_s')] = np.maximum(var.ymix[:,species.index('NH3_l_s')], 0) # cannot lose more than it has
        
        var.y = var.ymix * np.vstack( np.sum(var.y[:,atm.gas_indx], axis=1) )
        
        return var
