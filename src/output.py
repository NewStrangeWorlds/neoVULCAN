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

from chemistry_jax import chemdf, neg_achemjac
compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class Output(object):
    
    def __init__(self):
        
        output_dir, out_name, plot_dir = vulcan_cfg.output_dir, vulcan_cfg.out_name, vulcan_cfg.plot_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        if vulcan_cfg.use_save_movie == True:
            if not os.path.exists(vulcan_cfg.movie_dir): os.makedirs(vulcan_cfg.movie_dir)
        
        if os.path.isfile(output_dir+out_name):
            # Fix Python 3.x and 2.x.
            # try: input = raw_input
            # except NameError: pass
            # input("  The output file: " + str(out_name) + " already exists.\n"
            #           "  Press enter to overwrite the existing file,\n"
            #           "  or Ctrl-Z and Return to leave and choose a different out_name in vulcan_cfg.")
            
            print ('Warning... the output file: ' + str(out_name) + ' already exists.\n')
        
    def print_prog(self, var, para):
        indx_max = np.nanargmax(para.where_varies_most)
        print ('Elapsed time: ' +"{:.2e}".format(var.t) + ' || Step number: ' + str(para.count) + '/' + str(vulcan_cfg.count_max) ) 
        print ('longdy = ' + "{:.2e}".format(var.longdy) + '      || longdy/dt = ' + "{:.2e}".format(var.longdydt) + '  || dt = '+ "{:.2e}".format(var.dt) )      
        print ('from nz = ' + str(int(indx_max/ni)) + ' and ' + species[indx_max%ni])
        print ('------------------------------------------------------------------------' )
        
        
    def print_end_msg(self, var, para ): 
        print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time') 
        print (vulcan_cfg.out_name[:-4] + ' has successfully run to steady-state with ' + str(para.count) + ' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        print ('long dy = ' + f"{var.longdy:.6e}" + ' and long dy/dt = ' + f"{var.longdydt:.6e}" )
        
        print ('total atom loss:')
        for atom in vulcan_cfg.atom_list: 
            if atom not in getattr(vulcan_cfg, 'loss_ex', []):
                print (atom + ': ' + f"{var.atom_loss[atom]:.4e}" + ' ')
      
        print ('negative solution counter:')
        print (para.nega_count)
        print ('loss rejected counter:')
        print (para.loss_count)
        print ('delta rejected counter:')
        print (para.delta_count)
        if vulcan_cfg.use_shark == True: print ("It's a long journey to this shark planet. Don't stop bleeding.")
        print ('------ Live long and prosper \V/ ------') 

    def print_unconverged_msg(self, var, para, case): 
        
        if case == 2:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print (vulcan_cfg.out_name[:-4] + ' did not reach steady-state:')
            print ('long dy = ' + str(var.longdy) + ' and long dy/dt = ' + str(var.longdydt) )
            print ('Integration stopped before converged...\nMaximal allowed runtime exceeded ('+ f"{vulcan_cfg.runtime:.1e}" + ' sec)')
        elif case == 3:
            print ("After ------- %s seconds -------" % ( time.time()- para.start_time ) + ' s CPU time')
            print (vulcan_cfg.out_name[:-4] + ' did not reach steady-state:')
            print ('long dy = ' + str(var.longdy) + ' and long dy/dt = ' + str(var.longdydt) )
            print ('Integration stopped before converged...\nMaximal allowed steps exceeded ('+ str(vulcan_cfg.count_max) + ' steps)')
        
        print ('total atom loss:')
        for atom in vulcan_cfg.atom_list: 
            if atom not in getattr(vulcan_cfg, 'loss_ex', []):
                print (atom + ': ' + f"{var.atom_loss[atom]:.4e}" + ' ')
        print ('negative solution counter:')
        print (para.nega_count)
        print ('loss rejected counter:')
        print (para.loss_count)
        print ('delta rejected counter:')
        print (para.delta_count)
       
        if case not in (2, 3):
            raise RuntimeError(f"Unconverged case undefined (case={case})") # more robust than printing warning
        

    def save_cfg(self, dname):
        output_dir, out_name = vulcan_cfg.output_dir, vulcan_cfg.out_name
        if not os.path.exists(output_dir):
            print ('The output directory assigned in vulcan_cfg.py does not exist.')
            print( 'Directory ' , output_dir,  " created.")
            os.mkdir(output_dir)

        # copy the vulcan_cfg.py file
        with open('vulcan_cfg.py' ,'r') as f:
            cfg_str = f.read()
        with open(dname + '/' + output_dir + "cfg_" + out_name[:-3] + "txt", 'w') as f: f.write(cfg_str)
    
    def save_out(self, var, atm, para, dname): 
        output_dir, out_name = vulcan_cfg.output_dir, vulcan_cfg.out_name
        output_file = dname + '/' + output_dir + out_name
        
        if not os.path.exists(output_dir):
            print ('The output directory assigned in vulcan_cfg.py does not exist.')
            print( 'Directory ' , output_dir,  " created.")
            os.mkdir(output_dir)
            
        # convert lists into numpy arrays
        for key in var.var_evol_save:
            as_nparray = np.array(getattr(var, key))
            setattr(var, key, as_nparray)
        
        # plotting
        if vulcan_cfg.use_plot_evo == True: 
            self.plot_evo(var, atm)
        if vulcan_cfg.use_plot_end == True:
            self.plot_end(var, atm, para)
        else: plt.close()
        
        # making the save dict
        var_save = {'species':species, 'nr':nr}
        
        for key in var.var_save:
            var_save[key] = getattr(var, key)
        if vulcan_cfg.save_evolution == True:
            # slicing time-sequential data to reduce ouput filesize
            fq = vulcan_cfg.save_evo_frq
            for key in var.var_evol_save:
                as_nparray = getattr(var, key)[::fq]
                setattr(var, key, as_nparray)
                var_save[key] = getattr(var, key)

        with open(output_file, 'wb') as outfile:
            if vulcan_cfg.output_humanread == True: # human-readable form, less efficient 
                outfile.write(str({'variable': var_save, 'atm': vars(atm), 'parameter': vars(para)}))
            else:
                # the protocol must be <= 2 for python 2.X
                pickle.dump( {'variable': var_save, 'atm': vars(atm), 'parameter': vars(para) }, outfile, protocol=4)
                # how to add  'config': vars(vulcan_cfg) ?
        
            
    def plot_update(self, var, atm, para):
        
        images = []
        colors = ['b','g','r','c','m','y','k','orange','pink', 'grey',\
        'darkred','darkblue','salmon','chocolate','mediumspringgreen','steelblue','plum','hotpink']
        
        tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
        'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
        ,'CH2OH':'CH$_2$OH', 'NH3':'NH$_3$'}
        
        plt.figure('live mixing ratios')
        plt.ion()
        color_index = 0
        for color_index, sp in enumerate(vulcan_cfg.plot_spec):
            if sp in tex_labels: sp_lab = tex_labels[sp]
            else: sp_lab = sp
            if color_index == len(para.tableau20): # when running out of colors
                para.tableau20.append(tuple(np.random.rand(3)))
            if vulcan_cfg.plot_height == False:
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.pco/1.e6, color = para.tableau20[color_index], label=sp_lab)
                if vulcan_cfg.use_condense == True and sp in vulcan_cfg.condense_sp:
                    plt.plot(atm.sat_mix[sp], atm.pco/1.e6, color = para.tableau20[color_index], label=sp_lab + ' sat', ls='--')
                
                plt.gca().set_yscale('log')
                plt.gca().invert_yaxis()
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b/1.E6,vulcan_cfg.P_t/1.E6))
            else: # plotting with height
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.zmco/1.e5, color = para.tableau20[color_index], label=sp_lab)
                if vulcan_cfg.use_condense == True and sp in vulcan_cfg.condense_sp:
                    plt.plot(atm.sat_mix[sp], atm.zco[1:]/1.e5, color = para.tableau20[color_index], label=sp_lab + ' sat', ls='--')
                
                plt.ylim((atm.zco[0]/1e5,atm.zco[-1]/1e5))
                plt.ylabel("Height (km)")
                
            images.append((line,))
        
        plt.title(str(para.count)+' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        plt.gca().set_xscale('log')         
        plt.xlim(1.E-20, 1.)
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Mixing Ratios")
        plt.show(block=0)
        plt.pause(0.001)
        if vulcan_cfg.use_save_movie == True: 
            plt.savefig( vulcan_cfg.movie_dir+str(para.pic_count)+'.png', dpi=200)
            para.pic_count += 1
        plt.clf()
    
    def plot_flux_update(self, var, atm, para):
        
        images = []
        plt.ion()
        
        # fig.add_subplot(121) fig.add_subplot(122)
        
        line1, = plt.plot(np.sum(var.dflux_u,axis=1), atm.pico/1.e6, label='up flux')
        line2, = plt.plot(np.sum(var.dflux_d,axis=1), atm.pico/1.e6, label='down flux', ls='--', lw=1.2)
        line3, = plt.plot(np.sum(var.sflux,axis=1), atm.pico/1.e6, label='stellar flux', ls=':', lw=1.5)
            
        images.append((line1,line2))        
        
        plt.title(str(para.count)+' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        plt.gca().set_xscale('log')       
        plt.gca().set_yscale('log') 
        plt.gca().invert_yaxis() 
        plt.xlim(xmin=1.E-8)
        plt.ylim((atm.pico[0]/1.e6,atm.pico[-1]/1.e6))
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Diffusive flux")
        plt.ylabel("Pressure (bar)")
        plt.show(block=0)
        plt.pause(0.1)
        if vulcan_cfg.use_flux_movie == True: plt.savefig( 'plot/movie/flux-'+str(para.count)+'.jpg')
        
        plt.clf()
        
    def plot_end(self, var, atm, para):
        
        plot_dir = vulcan_cfg.plot_dir
        colors = ['b','g','r','c','m','y','k','orange','pink', 'grey',\
        'darkred','darkblue','salmon','chocolate','mediumspringgreen','steelblue','plum','hotpink']
        
        plt.figure('live mixing ratios')
        color_index = 0
        for sp in vulcan_cfg.plot_spec:
            if vulcan_cfg.plot_height == False:
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.pco/1.e6, color = colors[color_index], label=sp)
                plt.gca().set_yscale('log')
                plt.gca().invert_yaxis() 
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b/1.E6,vulcan_cfg.P_t/1.E6))
            else: # plotting with height
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.zmco/1.e5, color = colors[color_index], label=sp)
                plt.ylim((atm.zco[0]/1e5,atm.zco[0]/1e5))
                plt.ylabel("Height (km)")
            color_index +=1
                  
        plt.title(str(para.count)+' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        plt.gca().set_xscale('log')
        plt.xlim(1.E-20, 1.)
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Mixing Ratios")
        plt.savefig(plot_dir + 'mix.png')       
        if vulcan_cfg.use_live_plot == True:
            # plotting in the same window of real-time plotting
            plt.draw()
        elif vulcan_cfg.use_PIL == True: # plotting in a new window with PIL package            
            plot = Image.open(plot_dir + 'mix.png')
            plot.show()
            plt.close()
            
    def plot_evo(self, var, atm, plot_j=-1, plot_ymin=1e-20, dn=1):
        
        plot_spec = vulcan_cfg.plot_spec
        plot_dir = vulcan_cfg.plot_dir
        plt.figure('evolution')
        
        ymix_time = np.array(var.y_time/atm.n_0[:,np.newaxis])
        
        for i,sp in enumerate(vulcan_cfg.plot_spec):
            plt.plot(var.t_time[::dn], ymix_time[::dn,plot_j,species.index(sp)],c = plt.cm.rainbow(float(i)/len(plot_spec)),label=sp)

        plt.gca().set_xscale('log')       
        plt.gca().set_yscale('log') 
        plt.xlabel('time')
        plt.ylabel('mixing ratios')
        plt.ylim((plot_ymin,1.))
        plt.legend(frameon=0, prop={'size':14}, loc='best')
        plt.savefig(plot_dir + 'evo.png')
        if vulcan_cfg.use_PIL == True:
            plot = Image.open(plot_dir + 'evo.png')
            plot.show()
            plt.close()
        # else: plt.show(block = False)
    
    def plot_evo_inter(self, var, atm, plot_j=-1, plot_ymin=1e-20, dn=1):
        '''
        plot the evolution when the code is interrupted
        '''
        var.t_time = np.array(var.t_time)
        ymix_time = np.array(var.y_time/atm.n_0[:,np.newaxis])
        
        plot_spec = vulcan_cfg.plot_spec
        plot_dir = vulcan_cfg.plot_dir
        plt.figure('evolution')
    
        for i,sp in enumerate(vulcan_cfg.plot_spec):
            plt.plot(var.t_time[::dn], ymix_time[::dn,plot_j,species.index(sp)],c = plt.cm.rainbow(float(i)/len(plot_spec)),label=sp)

        plt.gca().set_xscale('log')       
        plt.gca().set_yscale('log') 
        plt.xlabel('time')
        plt.ylabel('mixing ratios')
        plt.ylim((plot_ymin,1.))
        plt.legend(frameon=0, prop={'size':14}, loc='best')
        plt.savefig(plot_dir + 'evo.png')
        if vulcan_cfg.use_PIL == True:
            plot = Image.open(plot_dir + 'evo.png')
            plot.show()
            plt.close()
    
    def plot_TP(self, atm):
        plot_dir = vulcan_cfg.plot_dir
        #plt.figure('TPK')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny() # ax1 and ax2 share y-axis

        if vulcan_cfg.plot_height == False:
            ax1.semilogy( atm.Tco, atm.pco/1.e6, c='black')
            ax2.loglog( atm.Kzz, atm.pico[1:-1]/1.e6, c='k', ls='--')
            plt.gca().invert_yaxis()
            plt.ylim((vulcan_cfg.P_b/1.E6,vulcan_cfg.P_t/1.E6))
            ax1.set_ylabel("Pressure (bar)")

        else: # plotting with height
            ax1.plot(atm.Tco, atm.zmco/1.e5, c='black')
            ax2.semilogx( atm.Kzz, atm.zmco[1:]/1.e5, c='k', ls='--') 
            ax1.set_ylabel("Height (km)")

        #plt.xlabel("Temperature (K)")
        ax1.set_xlabel("Temperature (K)")
        ax2.set_xlabel(r'K$_{zz}$ (cm$^2$s$^{-1}$)')
        
        plot_name = plot_dir + 'TPK.png'
        plt.savefig(plot_name)
        if vulcan_cfg.use_PIL == True:        
            plot = Image.open(plot_name)
            plot.show()
            # close the matplotlib window
            plt.close()
        else: plt.show(block = False)
        
        

## back up ###
# class SemiEU(ODESolver):
#     '''
#     class inheritance from ODEsolver for semi-implicit Euler solver
#     '''
#     def __init__(self):
#         ODESolver.__init__(self)
#
#     def solver(self, var, atm):
#         """
#         semi-implicit Euler solver (1st order)
#         """
#         y, ymix, h, k = var.y, var.ymix, var.dt, var.k
#         M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
#
#         diffdf = self.diffdf
#         jac_tot = self.jac_tot
#
#         df = chemdf(y,M,k).flatten() + diffdf(var, atm).flatten()
#         dfdy = jac_tot(var, atm)
#         aa = np.identity(ni*nz) - h*dfdy
#         aa = scipy.linalg.solve(aa,df)
#         aa = aa.reshape(y.shape)
#         y = y + aa*h
#
#         var.y = y
#         var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
#
#         return var
#
#     def step_ok(self, var, para, loss_eps = vulcan_cfg.loss_eps):
#         if np.all(var.y>=0) and np.amax( np.abs( np.fromiter(var.atom_loss.values(),float) - np.fromiter(var.atom_loss_prev.values(),float) ) )<loss_eps and para.delta<=rtol:
#             return True
#         else:
#             return False
#
#     def one_step(self, var, atm, para):
#
#         while True:
#            var = self.solver(var, atm)
#
#            # clipping small negative values and also calculating atomic loss (atom_loss)
#            var , para = self.clip(var, para, atm)
#
#            if self.step_ok(var, para): break
#            elif self.step_reject(var, para): break # giving up and moving on
#
#         return var, para
#
#     def step_size(self, var, para):
#         '''
#         PID control required for all semi-Euler like methods
#         '''
#         dt_var_min, dt_var_max, dt_min, dt_max = vulcan_cfg.dt_var_min, vulcan_cfg.dt_var_max, vulcan_cfg.dt_min, vulcan_cfg.dt_max
#         PItol = vulcan_cfg.PItol
#         dy, dy_prev, h = var.dy, var.dy_prev, var.dt
#
#         if dy == 0 or dy_prev == 0:
#             var.dt = np.minimum(h*2.,dt_max)
#             return var
#
#         if para.count > 0:
#
#             h_factor = (dy_prev/dy)**0.075 * (PItol/dy)**0.175
#             h_factor = np.maximum(h_factor, dt_var_min)
#             h_factor = np.minimum(h_factor, dt_var_max)
#             h *= h_factor
#             h = np.maximum(h, dt_min)
#             h = np.minimum(h, dt_max)
#
#         # store the adopted dt
#         var.dt = h
#
#         return var
#
#
# class SparSemiEU(SemiEU):
#     '''
#     class inheritance from SemiEU.
#     It is the same semi-implicit Euler solver except for utilizing sparse-matrix solvers
#     '''
#     def __init__(self):
#         SemiEU.__init__(self)
#
#     # override solver
#     def solver(self, var, atm):
#         """
#         sparse-matrix semi-implicit Euler solver (1st order)
#         """
#         y, ymix, h, k = var.y, var.ymix, var.dt, var.k
#         M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
#
#         diffdf = self.diffdf
#         jac_tot = self.jac_tot
#
#         df = chemdf(y,M,k).flatten() + diffdf(var, atm).flatten()
#         dfdy = jac_tot(var, atm)
#
#         aa = sparse.csc_matrix( np.identity(ni*nz) - h*dfdy )
#         aa = sparse.linalg.spsolve(aa,df)
#         aa = aa.reshape(y.shape)
#         y = y + aa*h
#
#         var.y = y
#         var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
#
#         return var
#
#
# ### back-up methods: extrapolation semi_implicit Euler ###
#         Kzz = atm.Kzz.copy()
#         vz = atm.vz.copy()
#         Tco = atm.Tco.copy()
#         mu, ms = atm.mu.copy(),  atm.ms.copy()
#         g = vulcan_cfg.g
#
#         r = 1. + 1./2.**0.5
#         c0 = 1./(r*var.dt)
#         dfdy = neg_achemjac(y, atm.M, var.k)
#         np.fill_diagonal(dfdy, c0 + np.diag(dfdy))
#         j_indx = []
#
#         for j in range(nz):
#             j_indx.append( np.arange(j*ni,j*ni+ni) )
#
#         for j in range(1,nz-1):
#             # excluding the buttom and the top cell
#             # at j level consists of ni species
#             dz_ave = 0.5*(dzi[j-1] + dzi[j])
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave
#
#         dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
#         # deposition velocity
#         if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
#
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
#
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]
#
#         return dfdy
#
#     def lhs_jac_fix_all_bot(self, var, atm):
#         """
#         directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
#         jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion)
#         Fixed all species BC: all species at bottom (y[0]) remains fixed
#         """
#         y = var.y.copy()
#         # TEST condensation excluding non-gaseous species
#         if vulcan_cfg.use_condense == True:
#             #ysum = np.sum(y[:,atm.gas_indx], axis=1)
#             ysum = np.sum(y, axis=1)
#         else: ysum = np.sum(y, axis=1)
#         # TEST condensation excluding non-gaseous species
#         dzi = atm.dzi.copy()
#         Kzz = atm.Kzz.copy()
#         Dzz = atm.Dzz.copy()
#         vz = atm.vz.copy()
#         alpha = atm.alpha.copy()
#         Tco = atm.Tco.copy()
#         mu, ms = atm.mu.copy(),  atm.ms.copy()
#         g = vulcan_cfg.g
#
#         Ti = atm.Ti.copy()
#         Hpi = atm.Hpi.copy()
#
#         r = 1. + 1./2.**0.5
#         c0 = 1./(r*var.dt)
#         dfdy = neg_achemjac(y, atm.M, var.k)
#         np.fill_diagonal(dfdy, c0 + np.diag(dfdy))
#         j_indx = []
#
#         for j in range(nz):
#             j_indx.append( np.arange(j*ni,j*ni+ni) )
#
#         for j in range(1,nz-1):
#             # excluding the buttom and the top cell
#             # at j level consists of ni species
#             dz_ave = 0.5*(dzi[j-1] + dzi[j])
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave
#
#             # [j_indx[j], j_indx[j]] has size ni*ni
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j]\
#             +1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
#             - Dzz[j-1]*(-1./Hpi[j-1]+ms*g/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) )
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
#             +1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
#             -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )
#
#         # deposition velocity (off with fixed all BC)
#         # if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
#
#         # Fix bottom BC
#         #print (dfdy[:, j_indx[0]])
#         dfdy[:, j_indx[0]] = 0.
#
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
#         +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )
#
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
#         - 1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
#                 -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] )
#
#         return dfdy
#
#     def lhs_jac_no_mol_fix_all_bot(self, var, atm):
#         """
#         directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
#         jacobian matrix for dn/dt + dphi/dz = P - L (WITHOUT molecular diffusion)
#         Fixed all species BC: all species at bottom (y[0]) remains fixed
#         """
#         y = var.y.copy()
#         # TEST condensation excluding non-gaseous species
#         if vulcan_cfg.use_condense == True:
#             #ysum = np.sum(y[:,atm.gas_indx], axis=1)
#             ysum = np.sum(y, axis=1)
#         else: ysum = np.sum(y, axis=1)
#         # TEST condensation excluding non-gaseous species
#         dzi = atm.dzi.copy()
#         Kzz = atm.Kzz.copy()
#         vz = atm.vz.copy()
#         Tco = atm.Tco.copy()
#         mu, ms = atm.mu.copy(),  atm.ms.copy()
#         g = vulcan_cfg.g
#
#         r = 1. + 1./2.**0.5
#         c0 = 1./(r*var.dt)
#         dfdy = neg_achemjac(y, atm.M, var.k)
#         np.fill_diagonal(dfdy, c0 + np.diag(dfdy))
#         j_indx = []
#
#         for j in range(nz):
#             j_indx.append( np.arange(j*ni,j*ni+ni) )
#
#         for j in range(1,nz-1):
#             # excluding the buttom and the top cell
#             # at j level consists of ni species
#             dz_ave = 0.5*(dzi[j-1] + dzi[j])
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave
#
#         #dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
#         # deposition velocity (off with fixed all BC)
#         # if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
#
#         # Fix bottom BC
#         dfdy[:, j_indx[0]] = 0.
#
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
#
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]
#
#         return dfdy
#
#     def lhs_jac_settling(self, var, atm):
#         """
#         directly constructing lhs = 1./(r*h)*sparse.identity(ni*nz) - dfdy
#         jacobian matrix for dn/dt + dphi/dz = P - L (including molecular diffusion and gravitation settling for particles)
#         zero-flux BC:  1st derivitive of y is zero
#         """
#         y = var.y.copy()
#         # TEST condensation excluding non-gaseous species
#         if vulcan_cfg.use_condense == True:
#             #ysum = np.sum(y[:,atm.gas_indx], axis=1)
#             ysum = np.sum(y, axis=1)
#         else: ysum = np.sum(y, axis=1)
#         # TEST condensation excluding non-gaseous species
#         dzi = atm.dzi.copy()
#         Kzz = atm.Kzz.copy()
#         Dzz = atm.Dzz.copy()
#         vz = atm.vz.copy()
#         vs = atm.vs.copy()
#         alpha = atm.alpha.copy()
#         Tco = atm.Tco.copy()
#         mu, ms = atm.mu.copy(),  atm.ms.copy()
#         g = vulcan_cfg.g
#
#         Ti = atm.Ti.copy()
#         Hpi = atm.Hpi.copy()
#
#         # c0 = 1./(r*h) where r = 1. + 1./2.**0.5
#         r = 1. + 1./2.**0.5
#         c0 = 1./(r*var.dt)
#         dfdy = neg_achemjac(y, atm.M, var.k)
#         np.fill_diagonal(dfdy, c0 + np.diag(dfdy))
#         j_indx = []
#
#         for j in range(nz):
#             j_indx.append( np.arange(j*ni,j*ni+ni) )
#
#         for j in range(1,nz-1):
#             # excluding the buttom and the top cell
#             # at j level consists of ni species
#             dz_ave = 0.5*(dzi[j-1] + dzi[j])
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j] -( (vz[j]>0)*vz[j] - (vz[j-1]<0)*vz[j-1] )/dz_ave
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Kzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) -( (vz[j]<0)*vz[j] )/dz_ave
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Kzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) +( (vz[j-1]>0)*vz[j-1] )/dz_ave
#
#             # [j_indx[j], j_indx[j]] has size ni*ni
#             dfdy[j_indx[j], j_indx[j]] -=  -1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/2. + Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/2. ) /ysum[j]\
#             +1./(2.*dz_ave)*( Dzz[j]*(-1./Hpi[j]+ms*g/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] ) \
#             - Dzz[j-1]*(-1./Hpi[j-1]+ms*g/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] ) )  -( (vs[j]>0)*vs[j] - (vs[j-1]<0)*vs[j-1] )/dz_ave
#             dfdy[j_indx[j], j_indx[j+1]] -= 1./dz_ave*( Dzz[j]/dzi[j]*(ysum[j+1]+ysum[j])/(2.*ysum[j+1]) ) \
#             +1./(2.*dz_ave)* Dzz[j]*(-1./Hpi[j]+ms*g/(Navo*kb*Ti[j])+alpha/Ti[j]*(Tco[j+1]-Tco[j])/dzi[j] )  -( (vs[j]<0)*vs[j] )/dz_ave
#             dfdy[j_indx[j], j_indx[j-1]] -= 1./dz_ave*( Dzz[j-1]/dzi[j-1]*(ysum[j-1]+ysum[j])/(2.*ysum[j-1]) ) \
#             -1./(2.*dz_ave)* Dzz[j-1]*(-1./Hpi[j-1]+ms*g/(Navo*kb*Ti[j-1])+alpha/Ti[j-1]*(Tco[j]-Tco[j-1])/dzi[j-1] )  +( (vs[j-1]>0)*vs[j-1] )/dz_ave
#
#         dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) -( (vz[0]>0)*vz[0] )/dzi[0]
#         dfdy[j_indx[0], j_indx[0]] -= -1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[0]) \
#         +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] )  -( (vs[0]>0)*vs[0] )/dzi[0]
#         # deposition velocity
#         if vulcan_cfg.use_botflux == True: dfdy[j_indx[0], j_indx[0]] -= -1.*atm.bot_vdep /dzi[0]
#
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Kzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) -( (vz[0]<0)*vz[0] )/dzi[0]
#         dfdy[j_indx[0], j_indx[1]] -= 1./(dzi[0])*(Dzz[0]/dzi[0]) * (ysum[1]+ysum[0])/(2.*ysum[1]) \
#         +1./(dzi[0])* Dzz[0]/2.*(-1./Hpi[0]+ms*g/(Navo*kb*Ti[0])+alpha/Ti[0]*(Tco[1]-Tco[0])/dzi[0] ) -( (vs[0]<0)*vs[0] )/dzi[0]
#
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2]) *(ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[nz-1]) +( (vz[-1]<0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[nz-1]] -= -1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[nz-1]) \
#         - 1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] ) +( (vs[-1]<0)*vs[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Kzz[nz-2]/dzi[nz-2])* (ysum[(nz-1)-1]+ysum[nz-1])/(2.*ysum[(nz-1)-1]) +( (vz[-1]>0)*vz[-1] )/dzi[-1]
#         dfdy[j_indx[nz-1], j_indx[(nz-1)-1]] -= 1./(dzi[nz-2])*(Dzz[nz-2]/dzi[nz-2]) *(ysum[nz-1]+ysum[nz-2])/(2.*ysum[(nz-1)-1]) \
#                 -1./(dzi[-1])* Dzz[-1]/2.*(-1./Hpi[-1]+ms*g/(Navo*kb*Ti[-1])+alpha/Ti[-1]*(Tco[-1]-Tco[-2])/dzi[-1] ) +( (vs[-1]>0)*vs[-1] )/dzi[-1]
#
#         return dfdy
#
#
#
#     def clip(self, var, para, atm, pos_cut = vulcan_cfg.pos_cut, nega_cut = vulcan_cfg.nega_cut):
#         '''
#         function to clip samll and negative values
#         and to calculate the particle loss
#         '''
#         y, ymix = var.y, var.ymix.copy()
#
#         para.small_y += np.abs(np.sum(y[np.logical_and(y<pos_cut, y>=0)]))
#         para.nega_y += np.abs(np.sum(y[np.logical_and(y>nega_cut, y<=0)]))
#         y[np.logical_and(y<pos_cut, y>=nega_cut)] = 0.
#
#         # Also setting y=0 when ymix<mtol
#         y[np.logical_and(ymix<self.mtol, y<0)] = 0.
#
#         var = self.loss(var)
#
#         # store y and ymix
#         # TEST condensation excluding non-gaseous species
#         if vulcan_cfg.use_condense == True:
#             #var.y, var.ymix = y, var.y/np.vstack(np.sum(var.y[:,atm.gas_indx],axis=1))
#             var.y, var.ymix = y, y/np.vstack(np.sum(y,axis=1))
#         else: var.y, var.ymix = y, y/np.vstack(np.sum(y,axis=1))
#         # TEST condensation excluding non-gaseous species
#
#         return var , para
#
#     def loss(self, data_var):
#
#         y = data_var.y
#         atom_list = vulcan_cfg.atom_list
#
#         # changed atom_tot to dictionary atom_sum
#         atom_sum = data_var.atom_sum
#
#         for atom in atom_list:
#             data_var.atom_sum[atom] = np.sum([compo[compo_row.index(species[i])][atom] * data_var.y[:,i] for i in range(ni)])
#             data_var.atom_loss[atom] = (data_var.atom_sum[atom] - data_var.atom_ini[atom])/data_var.atom_ini[atom]
#
#         return data_var
#
#     def step_ok(self, var, para, loss_eps = vulcan_cfg.loss_eps, rtol = vulcan_cfg.rtol):
#         if np.all(var.y>=0) and np.amax( np.abs( np.fromiter(var.atom_loss.values(),float) - np.fromiter(var.atom_loss_prev.values(),float) ) )<loss_eps and para.delta<=rtol:
#             return True
#         else:
#             return False
#
#     def step_reject(self, var, para, loss_eps = vulcan_cfg.loss_eps, rtol = vulcan_cfg.rtol):
#
#         if para.delta > rtol: # truncation error larger than the tolerence value
#             para.delta_count += 1
#
#         elif np.any(var.y < 0):
#             para.nega_count += 1
#             if vulcan_cfg.use_print_prog == True:
#                 self.print_nega(var,para) # print the info for the negative solutions (where y < 0)
#             # print input: y, t, count, dt
#
#
#         else: # meaning np.amax( np.abs( np.abs(y_loss) - np.abs(loss_prev) ) )<loss_eps
#             para.loss_count +=1
#             if vulcan_cfg.use_print_prog == True:
#                 self.print_lossBig(para)
#
#
#         var = self.reset_y(var) # reset y and dt to the values at previous step
#
#         if var.dt < vulcan_cfg.dt_min:
#             var.dt = vulcan_cfg.dt_min
#             var.y[var.y<0] = 0. # clipping of negative values
#             print ('Keep producing negative values! Clipping negative solutions and moving on!')
#             return True
#
#         return False
#
#     def reset_y(self, var, dt_reduc = vulcan_cfg.dt_var_min):
#         '''
#         reset y and reduce dt by dt_reduc
#         '''
#
#         # reset and store y and dt
#         var.y = var.y_prev
#         var.dt *= dt_reduc
#         # var.dt = np.maximum(var.dt, vulcan_cfg.dt_min)
#
#         return var
#
#     def print_nega(self, data_var, data_para):
#
#         nega_i = np.where(data_var.y<0)
#         print ('Negative y at time ' + str("{:.2e}".format(data_var.t)) + ' and step: ' + str(data_para.count) )
#         print ('Negative values:' + str(data_var.y[data_var.y<0]) )
#         print ('from levels: ' + str(nega_i[0]) )
#         print ('species: ' + str([species[_] for _ in nega_i[1]]) )
#         print ('dt= ' + str(data_var.dt))
#         print ('...reset dt to dt*0.2...')
#         print ('------------------------------------------------------------------')
#
#     def print_lossBig(self, para):
#
#         print ('partical conservation is violated too large (by numerical errors)')
#         print ('at step: ' + str(para.count))
#         print ('------------------------------------------------------------------')
#
#     def thomas_vec(a, b, c, d):
#         '''
#         Thomas vectorized solver, a b c d refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
#         d is a matrix
#         not used in this current version
#         '''
#         # number of equations
#         nf = len(a)
#         aa, bb, cc, dd = map(np.copy, (a, b, c, d))
#         # d needs to reshape
#         dd = dd.reshape(nf,-1)
#         #C' and D'
#         cp = [cc[0]/bb[0]]; dp = [dd[0]/bb[0]]
#         x = np.zeros((nf, np.shape(dd)[1]))
#
#         for i in range(1, nf-1):
#             cp.append( cc[i]/(bb[i] - aa[i]*cp[i-1]) )
#             dp.append( (dd[i] - aa[i]*dp[i-1])/(bb[i] - aa[i]*cp[i-1]) )
#
#         dp.append( (dd[(nf-1)] - aa[(nf-1)]*dp[(nf-1)-1])/(bb[(nf-1)] - aa[(nf-1)]*cp[(nf-1)-1]) ) # nf-1 is the last element
#         x[nf-1] = dp[nf-1]/1
#         for i in range(nf-2, -1, -1):
#             x[i] = dp[i] - cp[i]*x[i+1]
#
#         return x
#
#     ### photo-calculation starts from here
#
#     # def tot_cross(self, var):
# #     ''' compute the total cross section from all species '''
# #         for sp in vulcan_cfg.photo_sp:
# #             cross[sp]
#
#
#     def compute_tau(self, var, atm):
#         ''' compute the optical depth '''
#
#         # reset to zero
#         var.tau.fill(0)
#
#         for j in range(nz-1,-1,-1):
#
#             for sp in set.union(var.photo_sp,var.ion_sp):
#             # summing over all photo species
#                 var.tau[j] += var.y[j,species.index(sp)] * atm.dz[j] * var.cross[sp] # only the j-th laye
#
#             for sp in vulcan_cfg.scat_sp: # scat_sp are not necessary photo_sp, e.g. He
#                 var.tau[j] += var.y[j,species.index(sp)] * atm.dz[j] * var.cross_scat[sp]
#             # adding the layer above at the end of species loop
#             var.tau[j] += var.tau[j+1]
#
#     # Lines like chi = zeta_m**2*tran**2 - zeta_p**2 doing large np 2D array multiplication
#     # can be sped up with cython
#     def compute_flux(self, var, atm): # Vectorise this loop!
#         # change it to stagerred grids
#         # top: stellar flux
#         # bottom BC: zero upcoming flux
#
#         # Note!!! Matej's mu is defined in the outgoing hemisphere so his mu<0
#         # My cos[sl_angle] is always 0<=mu<=1
#         # Converting my mu to Matej's mu (e.g. 45 deg -> 135 deg)
#
#         mu_ang = -1.*np.cos(vulcan_cfg.sl_angle)
#         edd = vulcan_cfg.edd
#         tau = var.tau
#
#         # delta_tau (length nz) is used in the transmission function
#         delta_tau = tau - np.roll(tau,-1,axis=0) # np.roll(tau,-1,axis=0) are the upper layers
#         delta_tau = delta_tau[:-1]
#
#
#         # single-scattering albedo
#         nbins = len(var.bins)
#         tot_abs, tot_scat = np.zeros((nz, nbins)), np.zeros((nz, nbins))
#         for sp in var.photo_sp:
#             tot_abs += np.vstack(var.ymix[:,species.index(sp)])*var.cross[sp] # nz * nbins
#         for sp in vulcan_cfg.scat_sp:
#             tot_scat += np.vstack(var.ymix[:,species.index(sp)])*var.cross_scat[sp]
#
#         total = tot_abs + tot_scat
#
#         w0 = tot_scat  / (tot_abs + tot_scat) # 2D: nz * nbins
#         # tot_abs + tot_scat can be zero when certain gas (e.g. H2) does not exist
#
#         # Replace nan with zero and inf with very large numbers
#         w0 = np.nan_to_num(w0)
#
#         # to avoit w0=1
#         w0 = np.minimum(w0,1.-1.E-8)
#
#         # sflux: the direct beam; dflux: diffusive flux
#         ''' Beer's law for the intensity'''
#         var.sflux = var.sflux_top *  np.exp(-1.*tau/np.cos(vulcan_cfg.sl_angle) )
#         # converting the intensity to flux for the raditive transfer calculation
#         dir_flux = var.sflux*np.cos(vulcan_cfg.sl_angle) # multiplied by the zenith angle for calculating the diffuse flux
#
#         # scattering
#         # the transmission function (length nz)
#         if ag0 == 0: # to save memory
#             tran = np.exp( -1./edd *(1.- w0)**0.5 * delta_tau ) # 2D: nz * nbins
#             zeta_p = 0.5*( 1. + (1.-w0)**0.5 )
#             zeta_m = 0.5*( 1. - (1.-w0)**0.5 )
#             ll = -1.*w0/( 1./mu_ang**2 -1./edd**2 *(1.-w0) )
#             g_p = 0.5*( ll*(1./edd+1./mu_ang) )
#             g_m = 0.5*( ll*(1./edd-1./mu_ang) )
#
#         else:
#             tran = np.exp( -1./edd *( (1.- w0*ag0)*(1.- w0) )**0.5 * delta_tau )
#             zeta_p = 0.5*( 1. + ((1.-w0)/(1-w0*ag0))**0.5 )
#             zeta_m = 0.5*( 1. - ((1.-w0)/(1-w0*ag0))**0.5 )
#             ll = ( (1.-w0)*(1-w0*ag0) - 1.)/( 1./mu_ang**2 -1./edd**2 *(1.-w0)*(1-w0*ag0) )
#             g_p = 0.5*( ll*(1./edd+1/(mu_ang*(1.-w0*ag0))) + w0*ag0*mu_ang/(1.-w0*ag0)  )
#             g_m = 0.5*( ll*(1./edd-1/(mu_ang*(1.-w0*ag0))) - w0*ag0*mu_ang/(1.-w0*ag0)  )
#
#
#         # to avoit zero denominator
#         ll = np.minimum(ll, 1.e10)
#         ll = np.maximum(ll, -1.e10)
#
#
#         # 2D: nz * nbins
#         chi = zeta_m**2*tran**2 - zeta_p**2
#         xi = zeta_p*zeta_m*(1.-tran**2)
#         phi = (zeta_m**2-zeta_p**2)*tran
#
#         # 2D: nz * nbins
#         i_u = phi*g_p*dir_flux[:-1] - (xi*g_m+chi*g_p)*dir_flux[1:]
#         i_d = phi*g_m*dir_flux[1:] - (chi*g_m+xi*g_p)*dir_flux[:-1]
#         # sflux[1:] are all the layers above and sflux[:-1] are all the layers abelow
#
#         var.zeta_m = zeta_m
#         var.zeta_p = zeta_p
#         var.tran = tran
#
#
#
#         #starting recording time
#         #start_time = timeit.default_timer()
#
#
#         # propagating downward layer by layer and then upward
#         # var.dflux_d and var.dflux_p are defined at the interfaces (staggerred)
#         # the rest is defined in the center of the layer
#         for j in range(nz-1,-1,-1): # dflux_d goes from the second top interface (nz+1 interfaces)
#             var.dflux_d[j] = 1./chi[j]*(phi[j]*var.dflux_d[j+1] - xi[j]*var.dflux_u[j] + i_d[j]/mu_ang )
#         for j in range(1,nz+1):
#             var.dflux_u[j] = 1./chi[j-1]*(phi[j-1]*var.dflux_u[j-1] - xi[j-1]*var.dflux_d[j] + i_u[j-1]/mu_ang )
#
#
#         #print ("time passed...")
#         #print (timeit.default_timer() - start_time)
#
#
#         # old
#         # # the average intensity (not flux!) of the direct beam
# #         ave_int = 0.5*( var.sflux[:-1] + var.sflux[1:])
# #         tot_int = (ave_int + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:] + var.dflux_d[1:] + var.dflux_d[:-1]) )/edd
# #         # devided by the Eddington coefficient to recover the intensity
#
#
#         # the average flux from the direct beam
#         # !!! WITHOUT multiplied by the cos zenith angle (flux per unit area perpendicular to the direction of propagationat) !!!
#         ave_dir_flux = 0.5*( var.sflux[:-1] + var.sflux[1:])
#         # devided by the Eddington coefficient to recover the intensity then multiplied by 4pi to get the integrated flux
#         tot_flux = ave_dir_flux + 0.5*(var.dflux_u[:-1] + var.dflux_u[1:] + var.dflux_d[1:] + var.dflux_d[:-1])/edd
#
#
#         # ### Debug
#
#         #var.ave_int = ave_int
#
#         # var.ll = ll
#         # var.chi=chi
#         # var.phi=phi
#         # var.xi = xi
#         #
#         # var.i_u = i_u
#         # var.i_d = i_d
#         # var.w0 = w0
#         # var.tot_abs = tot_abs
#         # var.tot_scat = tot_scat
#         # var.tran = tran
#         # var.delta_tau = delta_tau
#
#         ### Debug
#         if np.any(tot_flux< -1.e-20):
#             print (tot_flux[tot_flux<-1.e-20])
#             raise IOError ('\nNegative diffusive flux! ')
#
#
#         # store the previous actinic flux into prev_aflux
#         var.prev_aflux = np.copy(var.aflux)
#         # converting to the actinic flux and storing the current flux
#         var.aflux = tot_flux / (hc/var.bins)
#         # the change of the actinic flux
#         var.aflux_change = np.nanmax( np.abs(var.aflux-var.prev_aflux)[var.aflux>vulcan_cfg.flux_atol]/var.aflux[var.aflux>vulcan_cfg.flux_atol] )
#
#         #print ('aflux change: ' + '{:.4E}'.format(var.aflux_change) )
#
#
#     def compute_J(self, var, atm): # the vectorised version
#         flux = var.aflux
#
#         #cross = var.cross
#         diss_cross = var.cross_J # use the key (sp, br) e.g. ("H2O", 1)
#
#         bins = var.bins
#         n_branch = var.n_branch
#
#         # reset to zeros every time
#         var.J_sp = dict([( (sp,bn) , np.zeros(nz)) for sp in var.photo_sp for bn in range(n_branch[sp]+1) ])
#
#         for sp in var.photo_sp:
#             # shape: flux (nz,nbin) cross (nbin)
#
#             # I want to parallelize this bit
#             # for n in range(var.nbin):
#             # I want to parallelize this bit
#
#             for nbr in range(1, n_branch[sp]+1):
#                 # axis=1 is to sum over all wavelength
#                 var.J_sp[(sp, nbr)] = np.sum( flux[:,:var.sflux_din12_indx] * diss_cross[(sp,nbr)][:var.sflux_din12_indx] * var.dbin1, axis=1)
#                 var.J_sp[(sp, nbr)] -= 0.5* (flux[:,0] * diss_cross[(sp,nbr)][0] + flux[:,var.sflux_din12_indx-1] * diss_cross[(sp,nbr)][var.sflux_din12_indx-1]) * var.dbin1
#                 var.J_sp[(sp, nbr)] += np.sum( flux[:,var.sflux_din12_indx:] * diss_cross[(sp,nbr)][var.sflux_din12_indx:] * var.dbin2, axis=1)
#                 var.J_sp[(sp, nbr)] -= 0.5* (flux[:,var.sflux_din12_indx] * diss_cross[(sp,nbr)][var.sflux_din12_indx] + flux[:,-1] * diss_cross[(sp,nbr)][-1]) * var.dbin2
#
#             # 0 is the total dissociation rate
#             # summing all branches
#             for nbr in range(1, n_branch[sp]+1):
#                 var.J_sp[(sp, 0)] += var.J_sp[(sp, nbr)]
#                 # incoperating J into rate coefficients
#                 if var.pho_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
#                     var.k[ var.pho_rate_index[(sp, nbr)]  ] = var.J_sp[(sp, nbr)] * vulcan_cfg.f_diurnal # f_diurnal = 0.5 for Earth; = 1 for tidally-loced planets
#
#
#
#     # Do Jion here
#     def compute_Jion(self, var, atm):
#         '''
#         compute the photoionisation rate
#         '''
#         flux = var.aflux
#         ion_cross = var.cross_Jion # use the key (sp, br) e.g. ("H2O", 1)
#
#         bins = var.bins
#         n_branch = var.ion_branch
#
#         # reset to zeros every time
#         var.Jion_sp = dict([( (sp,bn) , np.zeros(nz)) for sp in var.ion_sp for bn in range(n_branch[sp]+1) ])
#
#         for sp in var.ion_sp:
#             # shape: flux (nz,nbin) cross (nbin)
#
#             # convert to actinic flux *1/(hc/ld)
#             if wl_num == 0:
#                 for nbr in range(1, n_branch[sp]+1):
#                     # axis=1 is to sum over all wavelength
#                     var.Jion_sp[(sp, nbr)] = np.sum( flux[:,:var.sflux_din12_indx] * ion_cross[(sp,nbr)][:var.sflux_din12_indx] * var.dbin1, axis=1)
#                     var.Jion_sp[(sp, nbr)] -= 0.5* (flux[:,0] * ion_cross[(sp,nbr)][0]  + flux[:,var.sflux_din12_indx-1] * ion_cross[(sp,nbr)][var.sflux_din12_indx-1]) * var.dbin1
#                     var.Jion_sp[(sp, nbr)] += np.sum( flux[:,var.sflux_din12_indx:] * ion_cross[(sp,nbr)][var.sflux_din12_indx:] * var.dbin2, axis=1)
#                     var.Jion_sp[(sp, nbr)] -= 0.5* (flux[:,var.sflux_din12_indx] * ion_cross[(sp,nbr)][var.sflux_din12_indx]  + flux[:,-1] * ion_cross[(sp,nbr)][-1]) * var.dbin2
#
#             # end of the loop: for sp in var.photo_sp:
#
#             # 0 is the total dissociation rate
#             # summing all branches
#             for nbr in range(1, n_branch[sp]+1):
#                 var.Jion_sp[(sp, 0)] += var.Jion_sp[(sp, nbr)]
#                 # incoperating J into rate coefficients
#                 if var.ion_rate_index[(sp, nbr)] not in vulcan_cfg.remove_list:
#                     var.k[ var.ion_rate_index[(sp, nbr)]  ] = var.Jion_sp[(sp, nbr)] * vulcan_cfg.f_diurnal # f_diurnal = 0.5 for Earth; = 1 for tidally-loced planets
#
#
            
# class SemiEU(ODESolver):
#     '''
#     class inheritance from ODEsolver for semi-implicit Euler solver
#     '''
#     def __init__(self):
#         ODESolver.__init__(self)
#
#     def solver(self, var, atm):
#         """
#         semi-implicit Euler solver (1st order)
#         """
#         y, ymix, h, k = var.y, var.ymix, var.dt, var.k
#         M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
#
#         diffdf = self.diffdf
#         jac_tot = self.jac_tot
#
#         df = chemdf(y,M,k).flatten() + diffdf(var, atm).flatten()
#         dfdy = jac_tot(var, atm)
#         aa = np.identity(ni*nz) - h*dfdy
#         aa = scipy.linalg.solve(aa,df)
#         aa = aa.reshape(y.shape)
#         y = y + aa*h
#
#         var.y = y
#         var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
#
#         return var
#
#     def step_ok(self, var, para, loss_eps = vulcan_cfg.loss_eps):
#         if np.all(var.y>=0) and np.amax( np.abs( np.fromiter(var.atom_loss.values(),float) - np.fromiter(var.atom_loss_prev.values(),float) ) )<loss_eps and para.delta<=rtol:
#             return True
#         else:
#             return False
#
#     def one_step(self, var, atm, para):
#
#         while True:
#            var = self.solver(var, atm)
#
#            # clipping small negative values and also calculating atomic loss (atom_loss)
#            var , para = self.clip(var, para, atm)
#
#            if self.step_ok(var, para): break
#            elif self.step_reject(var, para): break # giving up and moving on
#
#         return var, para
#
#     def step_size(self, var, para):
#         '''
#         PID control required for all semi-Euler like methods
#         '''
#         dt_var_min, dt_var_max, dt_min, dt_max = vulcan_cfg.dt_var_min, vulcan_cfg.dt_var_max, vulcan_cfg.dt_min, vulcan_cfg.dt_max
#         PItol = vulcan_cfg.PItol
#         dy, dy_prev, h = var.dy, var.dy_prev, var.dt
#
#         if dy == 0 or dy_prev == 0:
#             var.dt = np.minimum(h*2.,dt_max)
#             return var
#
#         if para.count > 0:
#
#             h_factor = (dy_prev/dy)**0.075 * (PItol/dy)**0.175
#             h_factor = np.maximum(h_factor, dt_var_min)
#             h_factor = np.minimum(h_factor, dt_var_max)
#             h *= h_factor
#             h = np.maximum(h, dt_min)
#             h = np.minimum(h, dt_max)
#
#         # store the adopted dt
#         var.dt = h
#
#         return var
#
#
# class SparSemiEU(SemiEU):
#     '''
#     class inheritance from SemiEU.
#     It is the same semi-implicit Euler solver except for utilizing sparse-matrix solvers
#     '''
#     def __init__(self):
#         SemiEU.__init__(self)
#
#     # override solver
#     def solver(self, var, atm):
#         """
#         sparse-matrix semi-implicit Euler solver (1st order)
#         """
#         y, ymix, h, k = var.y, var.ymix, var.dt, var.k
#         M, dzi, Kzz = atm.M, atm.dzi, atm.Kzz
#
#         diffdf = self.diffdf
#         jac_tot = self.jac_tot
#
#         df = chemdf(y,M,k).flatten() + diffdf(var, atm).flatten()
#         dfdy = jac_tot(var, atm)
#
#         aa = sparse.csc_matrix( np.identity(ni*nz) - h*dfdy )
#         aa = sparse.linalg.spsolve(aa,df)
#         aa = aa.reshape(y.shape)
#         y = y + aa*h
#
#         var.y = y
#         var.ymix = var.y/np.vstack(np.sum(var.y,axis=1))
#
#         return var
#
#
