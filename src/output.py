import numpy as np
import matplotlib.pyplot as plt
import time, os, pickle

import vulcan_cfg
try: from PIL import Image
except ImportError:
    try: import Image
    except ImportError: vulcan_cfg.use_PIL = False

import build_atm
import chem_funs
from chem_funs import ni, nr

compo = build_atm.compo
compo_row = build_atm.compo_row
species = chem_funs.spec_list

class Output(object):
    
    def __init__(self):
        
        output_dir, out_name, plot_dir = vulcan_cfg.output_dir, vulcan_cfg.out_name, vulcan_cfg.plot_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        if vulcan_cfg.use_save_movie:
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
        if vulcan_cfg.use_shark: print ("It's a long journey to this shark planet. Don't stop bleeding.")
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
        if vulcan_cfg.use_plot_evo:
            self.plot_evo(var, atm)
        if vulcan_cfg.use_plot_end:
            self.plot_end(var, atm, para)
        else: plt.close()
        
        # making the save dict
        var_save = {'species':species, 'nr':nr}
        
        for key in var.var_save:
            var_save[key] = getattr(var, key)
        if vulcan_cfg.save_evolution:
            # slicing time-sequential data to reduce ouput filesize
            fq = vulcan_cfg.save_evo_frq
            for key in var.var_evol_save:
                as_nparray = getattr(var, key)[::fq]
                setattr(var, key, as_nparray)
                var_save[key] = getattr(var, key)

        data = {'variable': var_save, 'atm': vars(atm), 'parameter': vars(para)}
        if vulcan_cfg.output_humanread:
            with open(output_file, 'w') as outfile:
                outfile.write(str(data))
        else:
            with open(output_file, 'wb') as outfile:
                pickle.dump(data, outfile, protocol=4)
        
            
    def plot_update(self, var, atm, para):

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
            if not vulcan_cfg.plot_height:
                plt.plot(var.ymix[:,species.index(sp)], atm.pco/1.e6, color = para.tableau20[color_index], label=sp_lab)
                if vulcan_cfg.use_condense and sp in vulcan_cfg.condense_sp:
                    plt.plot(atm.sat_mix[sp], atm.pco/1.e6, color = para.tableau20[color_index], label=sp_lab + ' sat', ls='--')
                
                plt.gca().set_yscale('log')
                plt.gca().invert_yaxis()
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b/1.E6,vulcan_cfg.P_t/1.E6))
            else: # plotting with height
                plt.plot(var.ymix[:,species.index(sp)], atm.zmco/1.e5, color = para.tableau20[color_index], label=sp_lab)
                if vulcan_cfg.use_condense and sp in vulcan_cfg.condense_sp:
                    plt.plot(atm.sat_mix[sp], atm.zco[1:]/1.e5, color = para.tableau20[color_index], label=sp_lab + ' sat', ls='--')
                
                plt.ylim((atm.zco[0]/1e5,atm.zco[-1]/1e5))
                plt.ylabel("Height (km)")

        plt.title(str(para.count)+' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        plt.gca().set_xscale('log')         
        plt.xlim(1.E-20, 1.)
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Mixing Ratios")
        plt.show(block=0)
        plt.pause(0.001)
        if vulcan_cfg.use_save_movie: 
            plt.savefig( vulcan_cfg.movie_dir+str(para.pic_count)+'.png', dpi=200)
            para.pic_count += 1
        plt.clf()
    
    def plot_flux_update(self, var, atm, para):
        plt.ion()
        plt.plot(np.sum(var.dflux_u,axis=1), atm.pico/1.e6, label='up flux')
        plt.plot(np.sum(var.dflux_d,axis=1), atm.pico/1.e6, label='down flux', ls='--', lw=1.2)
        plt.plot(np.sum(var.sflux,axis=1), atm.pico/1.e6, label='stellar flux', ls=':', lw=1.5)

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
        if vulcan_cfg.use_flux_movie: plt.savefig( 'plot/movie/flux-'+str(para.count)+'.jpg')
        
        plt.clf()
        
    def plot_end(self, var, atm, para):
        
        plot_dir = vulcan_cfg.plot_dir
        colors = ['b','g','r','c','m','y','k','orange','pink', 'grey',\
        'darkred','darkblue','salmon','chocolate','mediumspringgreen','steelblue','plum','hotpink']
        
        plt.figure('live mixing ratios')
        color_index = 0
        for sp in vulcan_cfg.plot_spec:
            if not vulcan_cfg.plot_height:
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.pco/1.e6, color = colors[color_index], label=sp)
                plt.gca().set_yscale('log')
                plt.gca().invert_yaxis() 
                plt.ylabel("Pressure (bar)")
                plt.ylim((vulcan_cfg.P_b/1.E6,vulcan_cfg.P_t/1.E6))
            else: # plotting with height
                line, = plt.plot(var.ymix[:,species.index(sp)], atm.zmco/1.e5, color = colors[color_index], label=sp)
                plt.ylim((atm.zco[0]/1e5,atm.zco[-1]/1e5))
                plt.ylabel("Height (km)")
            color_index +=1
                  
        plt.title(str(para.count)+' steps and ' + str("{:.2e}".format(var.t)) + ' s' )
        plt.gca().set_xscale('log')
        plt.xlim(1.E-20, 1.)
        plt.legend(frameon=0, prop={'size':14}, loc=3)
        plt.xlabel("Mixing Ratios")
        plt.savefig(plot_dir + 'mix.png')       
        if vulcan_cfg.use_live_plot:
            # plotting in the same window of real-time plotting
            plt.draw()
        elif vulcan_cfg.use_PIL: # plotting in a new window with PIL package            
            plot = Image.open(plot_dir + 'mix.png')
            plot.show()
            plt.close()
            
    def plot_evo(self, var, atm, plot_j=-1, plot_ymin=1e-20, dn=1):
        plot_spec = vulcan_cfg.plot_spec
        plot_dir = vulcan_cfg.plot_dir
        t_time = np.array(var.t_time)
        ymix_time = np.array(var.y_time) / atm.n_0[:, np.newaxis]
        plt.figure('evolution')
        for i, sp in enumerate(plot_spec):
            plt.plot(t_time[::dn], ymix_time[::dn, plot_j, species.index(sp)],
                     c=plt.cm.rainbow(float(i)/len(plot_spec)), label=sp)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.xlabel('time')
        plt.ylabel('mixing ratios')
        plt.ylim((plot_ymin, 1.))
        plt.legend(frameon=0, prop={'size':14}, loc='best')
        plt.savefig(plot_dir + 'evo.png')
        if vulcan_cfg.use_PIL:
            plot = Image.open(plot_dir + 'evo.png')
            plot.show()
            plt.close()
    
    def plot_TP(self, atm):
        plot_dir = vulcan_cfg.plot_dir
        #plt.figure('TPK')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny() # ax1 and ax2 share y-axis

        if not vulcan_cfg.plot_height:
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
        if vulcan_cfg.use_PIL:        
            plot = Image.open(plot_name)
            plot.show()
            # close the matplotlib window
            plt.close()
        else: plt.show(block = False)

