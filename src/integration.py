import time
import numpy as np

import vulcan_cfg
import build_atm
import chem_funs
from chem_funs import ni
from phy_const import kb, Navo
from vulcan_cfg import nz
from condensation import Condensation

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class Integration:
    """Time-stepping until the stopping criteria (steady-state) is satisfied."""

    def __init__(self, odesolver, output):
        self.mtol = vulcan_cfg.mtol
        self.atol = vulcan_cfg.atol
        self.output = output

        self.odesolver = odesolver
        self.non_gas_sp = vulcan_cfg.non_gas_sp
        self.use_settling = vulcan_cfg.use_settling

        if vulcan_cfg.use_photo:
            self.update_photo_frq = vulcan_cfg.ini_update_photo_frq

        if vulcan_cfg.use_condense:
            self.non_gas_sp_index = [species.index(sp) for sp in self.non_gas_sp]
            self.condense_sp_index = [species.index(sp) for sp in vulcan_cfg.condense_sp]
            self.condensation = Condensation()
        
        
    def __call__(self, var, atm, para, make_atm):
        
        use_print_prog, use_live_plot = vulcan_cfg.use_print_prog, vulcan_cfg.use_live_plot
        self.loss_criteria = 0.0005

        while not self.stop(var, para, atm):

            var = self.backup(var)

            if vulcan_cfg.use_photo and var.longdy < vulcan_cfg.yconv_min*10. and var.longdydt < 1.e-6:
                self.update_photo_frq = vulcan_cfg.final_update_photo_frq
                if not para.switch_final_photo_frq:
                    print('update_photo_frq changed to ' + str(vulcan_cfg.final_update_photo_frq) + '\n')
                    para.switch_final_photo_frq = True

            if vulcan_cfg.use_photo and para.count % self.update_photo_frq == 0:
                self.odesolver.rt(var, atm)

            var, para = self.odesolver.one_step(var, atm, para)

            if vulcan_cfg.use_adapt_rtol and para.count % 10 == 0:
                if max([np.abs(loss) for loss in var.atom_loss.values()]) >= self.loss_criteria:
                    self.loss_criteria *= 2.
                    vulcan_cfg.rtol *= 0.75
                    vulcan_cfg.rtol = max(vulcan_cfg.rtol, vulcan_cfg.rtol_min)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_min:
                        print('rtol reduced to ' + str(vulcan_cfg.rtol))
                        print('------------------------------------------------------------------')

            if vulcan_cfg.use_adapt_rtol and para.count % 1000 == 0 and para.count > 0:
                if max([np.abs(loss) for loss in var.atom_loss.values()]) < 2e-4:
                    vulcan_cfg.rtol *= 1.25
                    vulcan_cfg.rtol = min(vulcan_cfg.rtol, vulcan_cfg.rtol_max)
                    if vulcan_cfg.rtol != vulcan_cfg.rtol_max:
                        print('rtol increased to ' + str(vulcan_cfg.rtol))
                        print('------------------------------------------------------------------')

            if vulcan_cfg.use_condense and var.t >= vulcan_cfg.start_conden_time and not para.fix_species_start:
                var = self.condensation.update(var, atm)

                if vulcan_cfg.fix_species and var.t > vulcan_cfg.stop_conden_time:
                    if not para.fix_species_start:
                        para.fix_species_start = True
                        vulcan_cfg.rtol = vulcan_cfg.post_conden_rtol
                        print("rtol changed to " + str(vulcan_cfg.rtol) + " after fixing the condensaed species.")
                        atm.vs *= 0
                        print("Turn off the settling velocity of all species")

                        var.fix_y = {}
                        for sp in vulcan_cfg.fix_species:
                            var.fix_y[sp] = np.copy(var.y[:, species.index(sp)])

                            if vulcan_cfg.fix_species_from_coldtrap_lev:
                                if sp in ('H2O_l_s', 'H2SO4_l', 'NH3_l_s', 'S8_l_s'):
                                    atm.conden_min_lev[sp] = nz - 1
                                else:
                                    sat_rho = atm.n_0 * atm.sat_mix[sp]
                                    conden_status = var.y[:, species.index(sp)] >= sat_rho
                                    atm.conden_status = conden_status

                                    if list(var.y[conden_status, species.index(sp)]):
                                        min_sat = np.amin(atm.sat_mix[sp][conden_status])
                                        atm.min_sat = min_sat
                                        conden_min_lev = np.where(atm.sat_mix[sp] == min_sat)[0].item()
                                        atm.conden_min_lev[sp] = conden_min_lev
                                        print(sp + " is now fixed from " + "{:.2e}".format(atm.pco[atm.conden_min_lev[sp]]/1e6) + " bar.")
                                    else:
                                        print(sp + " not condensed.")
                                        atm.conden_min_lev[sp] = 0

                if vulcan_cfg.use_relax:
                    if 'H2O' in vulcan_cfg.use_relax:
                        var = self.condensation.h2o_evap_relax(var, atm)
                    if 'NH3' in vulcan_cfg.use_relax:
                        var = self.condensation.nh3_evap_relax(var, atm)

            if para.count % vulcan_cfg.update_frq == 0:
                atm = self.update_mu_dz(var, atm, make_atm)
                atm = self.update_phi_esc(var, atm)

            if vulcan_cfg.use_condense:
                var.y[:, atm.gas_indx] = np.vstack(atm.n_0) * var.ymix[:, atm.gas_indx]
            else:
                var.y = np.vstack(atm.n_0) * var.ymix

            var = self.f_dy(var, para)
            var, para = self.save_step(var, para)
            var = self.odesolver.step_size(var, para)

            if use_print_prog and para.count % vulcan_cfg.print_prog_num == 0:
                self.output.print_prog(var, para)

            if vulcan_cfg.use_live_flux and vulcan_cfg.use_photo and para.count % vulcan_cfg.live_plot_frq == 0:
                self.output.plot_flux_update(var, atm, para)

            if use_live_plot and para.count % vulcan_cfg.live_plot_frq == 0:
                self.output.plot_update(var, atm, para)
            
            
        
    def backup(self, var):
        var.y_prev = np.copy(var.y)
        var.dy_prev = np.copy(var.dy)
        var.atom_loss_prev = var.atom_loss.copy()
        return var
        
    def update_mu_dz(self, var, atm, make_atm):
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
        if pref_indx != 0:
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
        if vulcan_cfg.use_moldiff:
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
        
        longdy = np.abs((y_time[count-1] - y_time[indx])/np.vstack(atm.n_0))
        longdy[ymix < mtol_conv] = 0
        longdy[y < atol] = 0
        
        # to get rid off non-convergent species, e.g. HC3N without sinks
        if 'conver_ignore' in dir(vulcan_cfg):
            for sp in vulcan_cfg.conver_ignore: longdy[:,species.index(sp)] = 0 # added 2023 
        
        if vulcan_cfg.use_condense:
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
        return False

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
        
    
