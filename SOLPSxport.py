"""
This class is used by routines in 'SOLPSxport_dr.py' to read experimental data,
read SOLPS data, and then calculate the updated radial transport coefficients to attempt
to match SOLPS to the experimental profiles

R.S. Wilcox, J.M. Canik and J.D. Lore 2020-2023
contact: wilcoxr@fusion.gat.com
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

import SOLPSutils as sut

plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.facecolor':'w'})
plt.rcParams.update({'mathtext.default': 'regular'})

eV = 1.60217662e-19


class SOLPSxport:

    def __init__(self, workdir, gfile_loc, impurity_list=['c']):
        """
        Inputs:
          workdir         Directory with the SOLPS outputs
          gfile_loc       location of corresponding g file
          impurity_list   List of all the impurity species included in the plasma simulation
        """

        # Try parsing gfile name
        workdir_short = None
        shot = None
        try:
            shot_loc_in_gfile_string = gfile_loc.rfind('g')
            shot = int(gfile_loc[shot_loc_in_gfile_string+1 : shot_loc_in_gfile_string+7])

            shot_ind_in_workdir = workdir.rfind(str(shot))
            if shot_ind_in_workdir > 0:
                workdir_short = workdir[shot_ind_in_workdir:]
        except:
            pass

        self.b2plot_ready = False
        if 'B2PLOT_DEV' in os.environ.keys():
            if os.environ['B2PLOT_DEV'] == 'ps':
                self.b2plot_ready = True

        self.data = {'workdir':workdir, 'workdir_short':workdir_short, 'gfile_loc': gfile_loc,
                     'expData':{'fitProfs':{}}, 'solpsData':{'profiles':{}},
                     'impurities':[imp.lower() for imp in impurity_list], 'shot':shot}
        self.timeid = None

    # ----------------------------------------------------------------------------------------        
    
    def getSOLPSlast10Profs(self, plotit = False, use_existing_last10 = False):
        """
        Generates and reads the .last10 files (produced by running '2d_profiles', which looks
        at the last 10 time steps in the run.log file)
        """
        working = str(self.data['workdir'])

        olddir = os.getcwd()
        os.chdir(working)
        if working[-1] != '/': working += '/'
        # Call 2d_profiles as default, so you don't accidentally look at old time steps
        if (not use_existing_last10) or (not os.path.isfile(working + 'ne3da.last10')):
            # Check to see if SOLPS has been sourced, skip this if not
            if 'B2PLOT_DEV' in os.environ.keys():
                print("Calling '2d_profiles' in directory: " + working)
                os.system('2d_profiles')

        if not os.path.isfile('ne3da.last10'):
            raise Exception('ERROR: Need to write to run.log and call "2d_profiles" first')

        rx, ne_ = sut.readProf('ne3da.last10')
        rx, dn_ = sut.readProf('dn3da.last10')
        rx, te_ = sut.readProf('te3da.last10')
        rx, ke_ = sut.readProf('ke3da.last10')
        rx, ti_ = sut.readProf('ti3da.last10')
        rx, ki_ = sut.readProf('ki3da.last10')

        # If last10.old files exist read these too
        if os.path.isfile('ne3da.last10.old'):
            rx, ne_old_ = sut.readProf('ne3da.last10.old')
            rx, dn_old_ = sut.readProf('dn3da.last10.old')
            rx, te_old_ = sut.readProf('te3da.last10.old')
            rx, ke_old_ = sut.readProf('ke3da.last10.old')
            rx, ti_old_ = sut.readProf('ti3da.last10.old')
            rx, ki_old_ = sut.readProf('ki3da.last10.old')
            found_old = True
        else:
            found_old = False
        
        os.chdir(olddir)

        # Cast everything as np array so it doesn't break later when performing math operations
        
        ne = np.array(ne_)
        dn = np.array(dn_)
        te = np.array(te_)
        ke = np.array(ke_)
        ti = np.array(ti_)
        ki = np.array(ki_)

        last10_dic = {'rx':rx,'ne':ne,'dn':dn,'te':te,'ke':ke,'ti':ti,'ki':ki}

        if found_old:
            last10_dic['ne_old'] = np.array(ne_old_)
            last10_dic['dn_old'] = np.array(dn_old_)
            last10_dic['te_old'] = np.array(te_old_)
            last10_dic['ke_old'] = np.array(ke_old_)
            last10_dic['ti_old'] = np.array(ti_old_)
            last10_dic['ki_old'] = np.array(ki_old_)

        self.data['solpsData']['last10'] = last10_dic
        
        if plotit:
            if 'psiSOLPS' in self.data['solpsData'].keys():
                psi = self.data['solpsData']['psiSOLPS']

                f, ax = plt.subplots(2, 3, sharex = 'all')
                ax[0, 0].plot(psi, ne / 1.0e19, 'r', lw = 2, label = 'SOLPS')
                ax[0, 0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
                ax[0, 0].grid('on')

                ax[1, 0].plot(psi, dn, '-ok', lw = 2)
                ax[1, 0].set_ylabel('D')
                ax[1, 0].set_xlabel('$\psi_N$')

                ax[0, 1].plot(psi, te / 1.0e3, 'r', lw = 2, label = 'SOLPS')
                ax[0, 1].set_ylabel('Te (keV)')

                ax[1, 1].plot(psi, ke, 'b', lw = 2)
                ax[1, 1].set_ylabel('$\chi_e$')
                ax[1, 1].set_xlabel('$\psi_N$')
                ax[1, 1].set_xlim([np.min(psi) - 0.01, np.max(psi) + 0.01])

                ax[0, 2].plot(psi, ti / 1.0e3, 'r', lw = 2, label = 'SOLPS')
                ax[0, 2].set_ylabel('Ti (keV)')

                ax[1, 2].plot(psi, ki, 'b', lw = 2)
                ax[1, 2].set_ylabel('$\chi_i$')
                ax[1, 2].set_xlabel('$\psi_N$')
                ax[0, 0].set_title('last10 profiles')
                plt.tight_layout()
                
                for i in range(2):
                    for j in range(3):
                        ax[i,j].grid('on')
                        
                plt.figure()
                plt.plot(psi, dn / ke, 'k', lw=3, label = 'D / $\chi_e$')
                plt.plot(psi, dn / ki, 'r', lw=3, label = 'D / $\chi_i$')
                plt.grid('on')
                ax[0, 0].set_xticks(np.arange(0.84, 1.05, 0.04))
                plt.legend(loc='best')
                        
            else:
                f, ax = plt.subplots(3, sharex = 'all')
                ax[0].plot(rx, ne*1e-19, '-kx', lw=2)
                ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
                ax[0].grid('on')
                
                ax[1].plot(rx, te, '-rx', lw=2, label = 'Te')
                ax[1].plot(rx, ti, '-bx', lw=2, label = 'Ti')
                ax[1].set_ylabel('T (eV)')
                ax[1].legend(loc='best')
                ax[1].grid('on')

                ax[2].plot(rx, dn, '-kx', lw=3, label = 'dn')
                ax[2].plot(rx, ke, '-gx', lw=3, label = 'ke')
                ax[2].plot(rx, ki, '-mx', lw=1, label = 'ki')
                ax[2].legend(loc='best')
                ax[2].set_ylabel('D or $\chi$')
                plt.grid('on')
                plt.tight_layout()
    
                ax[-1].set_xlabel('rx')
                
            plt.show(block = False)
        
    # ----------------------------------------------------------------------------------------
    
    def loadProfDBPedFit(self, profiles_file = None, shotnum = None,
                         timeid = None, runid = None, verbose = True):
        """
        Either (1) provide the location of the saved profiles file
                   (should have extension *.pkl)
            or (2) give info to retrieve it from MDSplus

            (1) can be written using getProfDBPedFit from SOLPSutils.py (easily reproduced manually if needed)
            (2) requires access to atlas.gat.com (probably running on Iris or Omega at GA)
        """
        
        if profiles_file is None:
            if verbose: print('Getting profile fit data from MDSplus server on atlas.gat.com')
            
            self.timeid = timeid
            self.data['expData']['fitVals'] = sut.getProfDBPedFit(shotnum, timeid, runid)
            
        else:
            if verbose: print('Reading profile fit data from saved file: ' + profiles_file)

            if profiles_file[-4:] == '.pkl':
                import pickle

                with open(profiles_file, 'rb') as f:
                    self.data['expData']['fitVals'] = pickle.load(f)

            else:
                print("ERROR: Cannot open ProfDB (Tom's Tools) profile file if not formatted as .pkl")
                # import json
                #
                # self.data['expData']['fitVals'] = json.load(open(profiles_file))

    # ----------------------------------------------------------------------------------------

    # def readMastData(self):
    #
    #     self.data['expData']['fitPsiProf'] =
    #     self.data['expData']['fitProfs']['neprof'] =
    #     self.data['expData']['fitProfs']['teprof'] =
    #
    #     try:
    #         self.data['expData']['fitProfs']['tiprof'] =

    # ----------------------------------------------------------------------------------------
    
    def load_pfile(self, pfile_loc, plotit = False):
        self.timeid = pfile_loc[pfile_loc.rfind('.')+1:]

        pfile_dict = sut.read_pfile(pfile_loc)
        self.data['expData']['pfile'] = pfile_dict

        self.data['expData']['fitPsiProf'] = pfile_dict['psinorm']
        self.data['expData']['fitProfs']['neprof'] = pfile_dict['ne(10^20/m^3)']
        self.data['expData']['fitProfs']['teprof'] = pfile_dict['te(KeV)']

        try:
            self.data['expData']['fitProfs']['tipsi'] = pfile_dict['psinorm']
            self.data['expData']['fitProfs']['tiprof'] = pfile_dict['ti(KeV)']
        except:
            print('No Ti data in pfile, defaulting to Ti = Te')
            self.data['expData']['fitProfs']['tiprof'] = self.data['expData']['fitProfs']['teprof']
        
        if plotit:
            solps_profs = self.data['solpsData']['last10']
            rne_SOLPS = solps_profs['rne']
            rte_SOLPS = solps_profs['rte']
            rti_SOLPS = solps_profs['rti']
            ne_SOLPS = solps_profs['ne']
            te_SOLPS = solps_profs['te']
            ti_SOLPS = solps_profs['ti']

            f, ax = plt.subplots(3, sharex = 'all')
            ax[0].plot(np.array(rne_SOLPS) + 1, ne_SOLPS * 1e-19, '-kx', lw=2, label='ne_SOLPS')
            ax[0].plot(self.data['expData']['fitPsiProf'], self.data['expData']['fitProfs']['neprof'],
                       '--r', lw = 2, label = 'ne')
            ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
            ax[0].legend(loc = 'best')
            ax[0].grid('on')

            ax[1].plot(np.array(rte_SOLPS) + 1, te_SOLPS, '-kx', lw = 2, label = 'Te_SOLPS')
            ax[1].plot(self.data['expData']['fitPsiProf'], self.data['expData']['fitProfs']['teprof'],
                       '--r', lw = 2, label = 'Te')
            ax[1].set_ylabel('Te (eV)')
            ax[1].legend(loc = 'best')
            ax[1].grid('on')
            
            ax[2].plot(np.array(rti_SOLPS) + 1, ti_SOLPS, '-kx', lw = 2, label = 'Ti_SOLPS')
            ax[2].plot(self.data['expData']['fitPsiProf'], self.data['expData']['fitProfs']['tiprof'],
                       '--r', lw = 2, label = 'Ti')
            ax[2].set_ylabel('Ti (eV)')
            ax[2].legend(loc = 'best')
            ax[2].grid('on')
        
            ax[-1].set_xlabel('$\psi_n$')
            ax[0].set_xlims([np.min(rte_SOLPS) - 0.01, np.max(rte_SOLPS) + 0.01])
            plt.show(block = False)

    # ----------------------------------------------------------------------------------------
    
    def populatePedFits(self, nemod='tanh', temod='tanh', ncmod='spl',
                        npsi=250, psinMax=None, plotit=False):
        """
        Get the fitted tanh profiles (need to have run loadProfDBPedFit already)
        Should nominally be able to handle other fit types, but only implemented 'tanh' and
        'tnh0' for electron profiles and 'tanh' and 'spl' for carbon density profiles so far

        Inputs:
          XXmod   Fit used for each parameter (only 'tanh' and 'tnh0' implemented and tested
                  for electron profiles so far, or 'tanh' and 'spl' for carbon density profiles)
          npsi    Number of points to populate full radial fit, from psin=0 (linearly distributed)
          psinMax End of radial grid to populate, in units of psin
                  (this defaults to be slightly larger than the SOLPS grid if nothing is given)
        """
        if (nemod not in ['tanh', 'tnh0']) or (temod not in ['tanh', 'tnh0']):
            print("ERROR: Cannot yet populate electron profile fits other than 'tanh' or 'tnh0'")
            return

        if psinMax is None:
            if 'psiSOLPS' in self.data['solpsData'].keys():
                psinMax = np.max(self.data['solpsData']['psiSOLPS']) + 0.001
            else:
                psinMax = 1.02

        psiProf = np.linspace(0, psinMax, npsi+1)
        
        self.data['expData']['fitPsiProf'] = psiProf
        self.data['expData']['fitProfs'] = {}
    
        # !!! Need to subtract shift from psiProf to move separatrix for power balance !!!

        # make ne profile
        necoef = self.data['expData']['fitVals']['ne' + nemod + 'psi']['y']
        neprof = sut.calcTanhMulti(necoef, psiProf)
        self.data['expData']['fitProfs']['neprof'] = neprof
            
        # make Te profile
        tecoef = self.data['expData']['fitVals']['te' + temod + 'psi']['y']
        teprof = sut.calcTanhMulti(tecoef, psiProf)
        self.data['expData']['fitProfs']['teprof'] = teprof

        # make Ti profile (already made, just populate)
        self.data['expData']['fitProfs']['tipsi'] = self.data['expData']['fitVals']['tisplpsi']['x']
        self.data['expData']['fitProfs']['tiprof'] = self.data['expData']['fitVals']['tisplpsi']['y']
        
        if ncmod == 'spl':
            zfzpsi = self.data['expData']['fitVals']['zfz1splpsi']['x']
            zfzprof = self.data['expData']['fitVals']['zfz1splpsi']['y']
            zfzfunc = interpolate.interp1d(zfzpsi, zfzprof, bounds_error = False,
                               fill_value = zfzprof[np.argmax(zfzpsi)])
            # extrapolate to psin>1 using highest value of psin available
        
            self.data['expData']['fitProfs']['ncprof'] = zfzfunc(psiProf) * neprof / 6
            
        elif ncmod == 'tanh':
            nccoef = self.data['expData']['fitVals']['nztanhpsi']['y']
            ncprof = sut.calcTanhMulti(nccoef,psiProf)
            self.data['expData']['fitProfs']['ncprof'] = ncprof * neprof / 6

        if np.any(neprof < 0) or np.any(teprof < 0):
            print("WARNING: Experimental profile is negative!!!")
            print("Either (a) modify profile by using a different fit, (b) set a " +
                  "decay length using 'enforce_decay_length', or (c) ignore the profile" +
                  " fit and set a constant diffusivity using 'flatSOLcoeffs'")
            
        if plotit:
            f, ax = plt.subplots(2, sharex = 'all')

            ax[0].plot(psiProf, neprof, '-b', lw = 2, label = 'n$_e$')
            ax[0].plot(psiProf, 6*self.data['expData']['fitProfs']['ncprof'], '--k', lw = 2, label = '6*n$_C$')
            ax[0].legend(loc='best')
            ax[0].set_ylabel('n (10$^{20}$ m$^{-3}$)')
            ax[0].grid('on')
    
            ax[1].plot(psiProf, teprof, '-r', lw = 2)
            ax[1].set_ylabel('T$_e$ (keV)')
            ax[1].grid('on')
            ax[1].set_xlabel('$\psi_n$')
            ax[1].set_xlim([0, psinMax+0.01])
            
            ax[0].set_title('Experimental Pedestal Fits')
            
            plt.show(block = False)

    # ----------------------------------------------------------------------------------------

    def load_ti(self, ti_fileloc = None, verbose = False):
        """
        Read Ti directly from a file if you have it (for MICER on DIII-D, but could be from elsewhere)
        """
        if verbose:
            print("Reading Ti data from: " + ti_fileloc)

        with open(ti_fileloc, 'r') as f:
            lines = f.readlines()

        psin = []
        ti = []

        for i, l in enumerate(lines):
            if l.startswith("#"):
                continue
            else:
                psin.append(float(l.strip().split()[0]))
                ti.append(float(l.strip().split()[1])/1.0e3)

        self.data['expData']['fitProfs']['tipsi'] = np.array(psin)
        self.data['expData']['fitProfs']['tiprof'] = np.array(ti)

    # ----------------------------------------------------------------------------------------

    def modify_ti(self, ratio_fileloc = None,
                  sol_points = None, max_psin = 1.1, decay_length = 0.015,
                  rad_loc_for_exp_decay = 1.0, reduce_ti = False, ti_min = 1, plotit = False):
        """
        Manually modify the Ti profile to be more realistic

        CER measures C+6, the temperature of which can differ significantly from
        the main ion species in the edge.

        Inputs:
          ratio_fileloc Location of file with a profile of the ratio of TD / TC
                        example file on GA clusters (Iris and Omega):
                        '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt'
          sol_points    Number of extra points to add in the SOL
          max_psi       sol_points will be evenly distributed between rad_loc_for_exp_decay
                        and max_psi
          decay_length  Decay length for exponential falloff imposed into SOL (in units of psin)
          reduce_ti     Use a profile from a single comparison case of T_D vs T_C+6
                        to reduce the "measured" value of T_D to be more realistic
          ti_min        Ti decays exponentially to this value (in eV)
        """

        tiexp = self.data['expData']['fitProfs']['tiprof']
        tiexppsi = self.data['expData']['fitProfs']['tipsi']

        r_cell = self.data['solpsData']['crLowerLeft']
        z_cell = self.data['solpsData']['czLowerLeft']
        psin = self.data['solpsData']['psiSOLPS']

        dr_flux = np.sqrt(np.diff(r_cell)**2 + np.diff(z_cell)**2)  # change in minor radial position (in m, but in flux r direction)

        ti_mod = tiexp.copy()
        xrad = tiexppsi.copy()

        if reduce_ti:

            print('Reducing T_D according to ratio of T_D / T_C from ' + ratio_fileloc)

            try:
                with open(ratio_fileloc, 'r') as f:
                    lines = f.readlines()

                psin_ratio = []
                T_DC_ratio = []  # The ratio T_D / T_C from 171558

                for line in lines:
                    elements = line.split()
                    if elements[0] != '#':
                        psin_ratio.append(np.float(elements[0]))
                        T_DC_ratio.append(np.float(elements[1]))

                T_ratio_fit = np.interp(tiexppsi, np.array(psin_ratio),
                                        np.array(T_DC_ratio), left=T_DC_ratio[0])
                # if > given range, chooses endpoint
                ti_reduced = tiexp * T_ratio_fit

            except FileNotFoundError:
                print("Can't retrieve T_D/T_C ratio file, not reducing Ti")
                ti_reduced = tiexp

            ti_mod = ti_reduced


        # Modify Ti profile to decay exponentially outside separatrix
        if decay_length is not None:
            outer_inds = np.where(tiexppsi >= rad_loc_for_exp_decay)[0]
            val_at_exp_decay_start = np.interp(rad_loc_for_exp_decay, tiexppsi, ti_mod)

            if sol_points is not None:
                xrad = np.delete(xrad, outer_inds)
                ti_mod = np.delete(ti_mod, outer_inds)

                extra_points = np.linspace(rad_loc_for_exp_decay, max_psin, sol_points + 1)
                xrad = np.append(xrad, extra_points)
                outer_inds = np.where(xrad >= rad_loc_for_exp_decay)[0]
                ti_mod = np.append(ti_mod, np.ones(sol_points + 1))

            ti_mod[outer_inds] = (val_at_exp_decay_start - ti_min * 1e-3) * \
                np.exp(-(xrad[outer_inds]-rad_loc_for_exp_decay) / decay_length) + ti_min * 1e-3

        if plotit:
            psi_TS = self.data['expData']['fitPsiProf']
            teexp = self.data['expData']['fitProfs']['teprof']

            plt.figure()
            plt.plot(psi_TS, teexp, 'g', lw=1, label = 'T$_e$ (TS)')
            plt.plot(tiexppsi, tiexp, '--sk', lw=2, label='T$_{C+6}$ (CER)')
            if reduce_ti:
                plt.plot(tiexppsi, ti_reduced, '-xr', ms=8, mew=2, lw=2,
                         label='T$_D$ (inferred)')
            plt.plot(xrad, ti_mod, '-ob', lw=3, label = 'Final T$_D$')
            plt.xlabel('$\psi_n$')
            plt.ylabel('T$_i$ (keV)')
            plt.legend(loc='best')
            plt.grid('on')
            plt.show(block=False)

        self.data['expData']['fitProfs']['ti_mod_psi'] = xrad
        self.data['expData']['fitProfs']['ti_mod'] = ti_mod

    # ----------------------------------------------------------------------------------------

    def modify_te(self, sol_points = None, max_psin = 1.1, decay_length = 0.015,
                  rad_loc_for_exp_decay = 1.0, te_min = 1, plotit = False):
        """
        Manually modify the Te profile. One way to handle fits with positive gradients or other
        issues in the far SOL.

        Inputs:
          sol_points    Number of extra points to add in the SOL
          max_psi       sol_points will be evenly distributed between rad_loc_for_exp_decay
                        and max_psi
          decay_length  Decay length for exponential falloff imposed into SOL (in units of psin)
          te_min        Te decays exponentially to this value (in eV)
        """

        teexp = self.data['expData']['fitProfs']['teprof']
        teexppsi = self.data['expData']['fitPsiProf']

        r_cell = self.data['solpsData']['crLowerLeft']
        z_cell = self.data['solpsData']['czLowerLeft']
        psin = self.data['solpsData']['psiSOLPS']

        dr_flux = np.sqrt(np.diff(r_cell)**2 + np.diff(z_cell)**2)  # change in minor radial position (in m, but in flux r direction)

        te_mod = teexp.copy()
        xrad = teexppsi.copy()

        # Modify profile to decay exponentially outside separatrix
        if decay_length is not None:
            outer_inds = np.where(teexppsi >= rad_loc_for_exp_decay)[0]
            val_at_exp_decay_start = np.interp(rad_loc_for_exp_decay, teexppsi, te_mod)

            if sol_points is not None:
                xrad = np.delete(xrad, outer_inds)
                te_mod = np.delete(te_mod, outer_inds)

                extra_points = np.linspace(rad_loc_for_exp_decay, max_psin, sol_points + 1)
                xrad = np.append(xrad, extra_points)
                outer_inds = np.where(xrad >= rad_loc_for_exp_decay)[0]
                te_mod = np.append(te_mod, np.ones(sol_points + 1))

            te_mod[outer_inds] = (val_at_exp_decay_start - te_min * 1e-3) * \
                np.exp(-(xrad[outer_inds]-rad_loc_for_exp_decay) / decay_length) + te_min * 1e-3

        self.data['expData']['fitProfs']['te_mod_psi'] = xrad
        self.data['expData']['fitProfs']['te_mod'] = te_mod
    # ----------------------------------------------------------------------------------------

    def modify_ne(self, sol_points = None, max_psin = 1.1, decay_length = 0.015,
                  rad_loc_for_exp_decay = 1.0, ne_min = 1e18, plotit = False):
        """
        Manually modify the ne profile. One way to handle fits with positive gradients or other
        issues in the far SOL.

        Inputs:
          sol_points    Number of extra points to add in the SOL
          max_psi       sol_points will be evenly distributed between rad_loc_for_exp_decay
                        and max_psi
          decay_length  Decay length for exponential falloff imposed into SOL (in units of psin)
          ne_min        ne decays exponentially to this value (in m^-3)
        """

        neexp = self.data['expData']['fitProfs']['neprof']
        neexppsi = self.data['expData']['fitPsiProf']

        r_cell = self.data['solpsData']['crLowerLeft']
        z_cell = self.data['solpsData']['czLowerLeft']
        psin = self.data['solpsData']['psiSOLPS']

        dr_flux = np.sqrt(np.diff(r_cell)**2 + np.diff(z_cell)**2)  # change in minor radial position (in m, but in flux r direction)

        ne_mod = neexp.copy()
        xrad = neexppsi.copy()

        # Modify profile to decay exponentially outside separatrix
        if decay_length is not None:
            outer_inds = np.where(neexppsi >= rad_loc_for_exp_decay)[0]
            val_at_exp_decay_start = np.interp(rad_loc_for_exp_decay, neexppsi, ne_mod)

            if sol_points is not None:
                xrad = np.delete(xrad, outer_inds)
                ne_mod = np.delete(ne_mod, outer_inds)

                extra_points = np.linspace(rad_loc_for_exp_decay, max_psin, sol_points + 1)
                xrad = np.append(xrad, extra_points)
                outer_inds = np.where(xrad >= rad_loc_for_exp_decay)[0]
                ne_mod = np.append(ne_mod, np.ones(sol_points + 1))

            ne_mod[outer_inds] = (val_at_exp_decay_start - ne_min * 1e-20) * \
                np.exp(-(xrad[outer_inds]-rad_loc_for_exp_decay) / decay_length) + ne_min * 1e-20

        self.data['expData']['fitProfs']['ne_mod_psi'] = xrad
        self.data['expData']['fitProfs']['ne_mod'] = ne_mod

    # ----------------------------------------------------------------------------------------

    def enforce_decay_length(self, prof_choice, sol_points = None, max_psin = 1.1, decay_length = 0.015,
                             rad_loc_for_exp_decay = 1.0, min_val = 1, plotit = False):
        """
        Enforce an exponential decay length in the SOL beyond a designated position

        Ensures smooth profiles, and can avoid dealing with density shoulder or unphysical measurements

        Inputs:
          prof_choice    Choose from Te, ne, or Ti

        """
        pass

    # ----------------------------------------------------------------------------------------
        
    def getProfsOMFIT(self, prof_folder, prof_filename_prefix, min_npsi = 100,
                      psiMax=1.05, plotit = False):
        """
        Reads the prof files from OMFIT fits (saved as prof*.txt files) and produce fits
          These were made using 'OMFIT_tools.py'
          (they're just text files with individual fitted profiles)
        
        Only plots if the profiles are remapped
        """
        # Read in the prof*.txt files
        
        if os.path.isfile(os.path.join(prof_folder,prof_filename_prefix + '_T_D.txt')):
            profs = ['n_e', 'T_e', 'n_12C6', 'T_D']
        else:
            profs = ['n_e', 'T_e', 'n_12C6', 'T_12C6']
    
        psin = {}
        vals = {}
        for fit in profs:
            psin_read, vals_read = sut.readProf(prof_filename_prefix + '_' + fit + '.txt',
                                                wdir = prof_folder)
            psin[fit] = np.array(psin_read)
            vals[fit] = np.array(vals_read)
        
        if 'fitVals' not in self.data['expData']:
            self.data['expData']['fitVals'] = {'nedatpsi':{}, 'zfz1datpsi':{}, 'tisplpsi':{}}
        
        # Set 'fitProfs' values (linearly interpolate if psi locations from prof files are no good)
        
        if (all(psin[f] == psin[fit]) for f in profs) and len(psin[fit]) >= min_npsi:
            
            # Deuterium profiles
            
            self.data['expData']['fitPsiProf'] = psin[fit]
            self.data['expData']['fitProfs'] = {'neprof': vals['n_e'] / 1.0e20,
                                                'teprof': vals['T_e'] / 1.0e3}
            # keep units consistent with Tom's tools

            self.data['expData']['fitProfs']['tipsi'] = psin[fit]
            if 'T_D' in vals.keys():
                self.data['expData']['fitProfs']['tiprof'] = vals['T_D'] / 1.0e3
            else:   # assume T_D = T_C
                self.data['expData']['fitProfs']['tiprof'] = vals['T_12C6'] / 1.0e3
            
            # Carbon profiles
            
            if 'c' in self.data['impurities']:
                self.data['expData']['fitVals']['nedatpsi']['x'] = psin[fit]
                self.data['expData']['fitVals']['nedatpsi']['y'] = vals['n_e'] / 1.0e20
                
                self.data['expData']['fitVals']['zfz1datpsi']['x'] = psin[fit]
                self.data['expData']['fitVals']['zfz1datpsi']['y'] = 6*vals['n_12C6']/vals['n_e']

                self.data['expData']['fitProfs']['ncprof'] = vals['n_12C6']
            
        else:
            
            psiProf = np.linspace(0, psiMax, min_npsi + 1)
    
            self.data['expData']['fitPsiProf'] = psiProf
            vals_interp = {}
            for fit in profs:
                fitfunc = interpolate.interp1d(psin[fit], vals[fit])
                vals_interp[fit] = fitfunc(psiProf)
            
            if plotit:
                f, ax = plt.subplots(len(profs), sharex = 'all')
                for i, fit in enumerate(profs):
                    ax[i].plot(psin[fit], vals[fit], 'ro', lw = 2, label = 'Profs file')
                    ax[i].plot(psiProf, vals_interp[fit], 'kx', lw = 2, label = 'Interpolated')
                    ax[i].set_ylabel(fit)
                    ax[i].grid('on')
                    
                ax[-1].set_xlabel('$\psi_n$')
                ax[0].legend(loc='best')
                plt.show(block=False)

    # ----------------------------------------------------------------------------------------

    def calcPsiVals(self, plotit = False, dsa = None, b2mn = None, geo = None, verbose=True):
        """
        Call b2plot to get the locations of each grid cell in psin space

        Saves the values to dictionaries in self.data['solpsData']

        Find grid corners first:
          0: lower left
          1: lower right
          2: upper left
          3: upper right

        Average location of cells 0 and 2 for middle of 'top' surface, 
        which is the top looking at outboard midplane
        Don't average over whole cell, dR << dZ at outboard midplane 
        and surface has curvature, so psin will be low

        jxa = poloidal cell index for the outer midplane
        crx = radial coordinate corner of grid [m]
        cry = vertical coordinate corner of grid [m]
        writ = write b2plot.write file
        f.y = plot against y
        """

        wdir = self.data['workdir']

        try:
            if dsa is None:
                dsa = sut.read_dsa('dsa')
            if geo is None:
                geo = sut.read_b2fgmtry('../baserun/b2fgmtry')
            if b2mn is None:
                b2mn = sut.scrape_b2mn("b2mn.dat")                

            # todo: add guess of jxa based on topology
            if not "jxa" in b2mn.keys():
                print('Warning: jxa not set in b2mn.dat! Will cause errors!')

            crLowerLeft = geo['crx'][b2mn['jxa']+1,:,0]
            crUpperLeft = geo['crx'][b2mn['jxa']+1,:,2]
            czLowerLeft = geo['cry'][b2mn['jxa']+1,:,0]
            czUpperLeft = geo['cry'][b2mn['jxa']+1,:,2]               
        except:
            if verbose:
                print('  Failed to read geometry files directly, trying b2plot')
            if not self.b2plot_ready:
                sut.set_b2plot_dev(verbose=verbose)
                self.b2plot_ready = True

            try:
                dsa, crLowerLeft = sut.B2pl('0 crx writ jxa f.y', wdir = wdir)
            except Exception as err:
                print('Exiting from calcPsiVals')
                raise err
        
            # Only 2 unique psi values per cell, grab 0 and 2
            dummy, crUpperLeft = sut.B2pl('2 crx writ jxa f.y', wdir = wdir)  # all x inds are the same
            dummy, czLowerLeft = sut.B2pl('0 cry writ jxa f.y', wdir = wdir)
            dummy, czUpperLeft = sut.B2pl('2 cry writ jxa f.y', wdir = wdir)
            
        ncells = len(czLowerLeft)

        g = sut.loadg(self.data['gfile_loc'])
        psiN = (g['psirz'] - g['simag']) / (g['sibry'] - g['simag'])

        dR = g['rdim'] / (g['nw'] - 1)
        dZ = g['zdim'] / (g['nh'] - 1)

        gR = []
        for i in range(g['nw']):
            gR.append(g['rleft'] + i * dR)

        gZ = []
        for i in range(g['nh']):
            gZ.append(g['zmid'] - 0.5 * g['zdim'] + i * dZ)

        gR = np.array(gR)
        gZ = np.array(gZ)

        R_solps_top = 0.5 * (np.array(crLowerLeft) + np.array(crUpperLeft))
        Z_solps_top = 0.5 * (np.array(czLowerLeft) + np.array(czUpperLeft))

        psiNinterp = interpolate.interp2d(gR, gZ, psiN, kind = 'cubic')

        psi_solps = np.zeros(ncells)
        for i in range(ncells):
            psi_solps_LL = psiNinterp(crLowerLeft[i], czLowerLeft[i])
            psi_solps_UL = psiNinterp(crUpperLeft[i], czUpperLeft[i])
            psi_solps[i] = np.mean([psi_solps_LL,psi_solps_UL])

        self.data['solpsData']['crLowerLeft'] = np.array(crLowerLeft)
        self.data['solpsData']['czLowerLeft'] = np.array(czLowerLeft)
        self.data['solpsData']['dsa'] = np.array(dsa)
        self.data['solpsData']['psiSOLPS'] = np.array(psi_solps)

        if plotit:
            psiN_range = [np.min(psi_solps), np.max(psi_solps)]

            psiN_copy = psiN.copy()
            psiN_copy[psiN > psiN_range[1]] = np.nan
            psiN_copy[psiN < psiN_range[0]] = np.nan
            psin_masked = np.ma.masked_invalid(psiN_copy)

            plt.figure()
            plt.contourf(gR, gZ, psin_masked, levels=[psiN_range[0], 1, psiN_range[1]], colors = ['b', 'b'])
            # plt.pcolormesh(gR, gZ, psin_masked, cmap = 'inferno')
            # plt.colorbar(ticks = [0.25,0.5,0.75,1])
            plt.plot(g['rlim'], g['zlim'], 'k', lw = 2)
            # plt.contour(gR, gZ, psiN, [1], colors='k')
            # plt.contour(gR, gZ, psiN, [psiN_range[0]], colors='r', linestyles='dashed')
            # plt.contour(gR, gZ, psiN, [psiN_range[1]], colors='r', linestyles='dashed')
            gfile_name = self.data['gfile_loc'][self.data['gfile_loc'].rfind('/')+1:]
            plt.title(gfile_name)
            plt.axis('equal')
            plt.xlabel('R (m)')
            plt.ylabel('Z (m)')
            plt.xticks([1.0, 1.5, 2.0, 2.5])
            plt.yticks(np.arange(-1.5, 1.6, 0.5))
            plt.xlim([np.min(gR), np.max(gR)])
            plt.ylim([np.min(gZ), np.max(gZ)])
            plt.plot(R_solps_top, Z_solps_top, 'g', lw=3)
            plt.plot([1.94, 1.94], [-1.5, 1.5], ':k', lw=1)  # Thomson laser path
            plt.tight_layout()

            plt.figure()
            plt.plot(R_solps_top, psi_solps, 'k', lw = 2)
            plt.xlabel('R at midplane (m)')
            plt.ylabel('$\psi_N$')

            plt.show(block = False)
    
    # ----------------------------------------------------------------------------------------
    
    def getSOLPSfluxProfs(self, plotit = False, b2mn = None, geo = None, state = None, dsa = None, xport = None):
        """
        Tries to get flux profiles from solps output, falls back to 
        Calls b2plot to get the particle flux profiles
        """

        try:
            if dsa is None:
                dsa = sut.read_dsa('dsa')
            if geo is None:
                geo = sut.read_b2fgmtry('../baserun/b2fgmtry')
            if b2mn is None:
                b2mn = sut.scrape_b2mn("b2mn.dat")                
            if state is None:
                state = sut.read_b2fstate("b2fstate")
            if xport is None:
                xport = sut.read_transport_files(".", dsa=dsa, geo=geo, state=state)
            
            sy = sut.avg_like_b2plot(geo['gs'][b2mn['jxa']+1,:,1])        
            z = np.ones((geo['nx']+2,geo['ny']+2,state['ns']))
            for i in range(state['ns']):
                z[:,:,i] = z[:,:,i]*state['zamin'][i]        
            fluxTot = sut.avg_like_b2plot(np.sum(state['fna'][b2mn['jxa']+1,:,1,:]*z[b2mn['jxa']+1,:,:],axis=1))/sy
            fluxD   = sut.avg_like_b2plot(state['fna'][b2mn['jxa']+1,:,1,1])/sy
            fluxConv = sut.avg_like_b2plot(np.sum(xport['vlay'][:,:]*state['na'][b2mn['jxa']+1,:,:]*z[b2mn['jxa']+1,:,:],axis=1))/sy
                
            na = np.sum(state['na'][b2mn['jxa']+1,:,:],axis=1)        
            qe = sut.avg_like_b2plot(state['fhe'][b2mn['jxa']+1,:,1])/sy
            qi = sut.avg_like_b2plot(state['fhi'][b2mn['jxa']+1,:,1])/sy
            x_fTot = dsa
        except:
            print('  Falling back to b2plot calls to get fluxes')

            if not self.b2plot_ready:
                sut.set_b2plot_dev()
                self.b2plot_ready = True

            # x variable is identical for all of these
            x_fTot, fluxTot = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ jxa f.y")
            x_fTot, fluxD = sut.B2pl("fnay 1 zsel sy m/ writ jxa f.y")
            dummy, fluxConv = sut.B2pl("na za m* vlay m* 0 0 sumz writ jxa f.y")
            dummy, na = sut.B2pl("na 0 0 sumz writ jxa f.y")
            dummy, qe = sut.B2pl("fhey sy m/ writ jxa f.y")
            dummy, qi = sut.B2pl("fhiy sy m/ writ jxa f.y")
        
            for c in [fluxTot, fluxConv]:
                if not c:
                    print("WARNING: Variable not populated by b2plot in getSOLPSfluxProfs")
                    print("  Make sure ncl_ncar and netcdf modules are loaded")
                    break

        self.data['solpsData']['profiles']['x_fTot'] = np.array(x_fTot)
        self.data['solpsData']['profiles']['fluxTot'] = np.array(fluxTot)
        self.data['solpsData']['profiles']['fluxD'] = np.array(fluxD)
        self.data['solpsData']['profiles']['fluxConv'] = np.array(fluxConv)
        self.data['solpsData']['profiles']['na'] = np.array(na)
        self.data['solpsData']['profiles']['qe'] = np.array(qe)
        self.data['solpsData']['profiles']['qi'] = np.array(qi)

        if plotit:
                
            # Check electron density from last10 profs for consistency
            ne_last10 = self.data['solpsData']['last10']['ne']
            rx_last10 = self.data['solpsData']['last10']['rx']  # very slightly different...

            f, ax = plt.subplots(2, sharex = 'all')

            ax[0].plot(rx_last10, ne_last10, '-kx', lw = 1, label = 'ne_last10')
            ax[0].plot(x_fTot, na, '--r*', lw=2, label = 'na')
            ax[0].set_ylabel('n (m$^{-3}$)')
            ax[0].legend(loc='best')
            ax[0].grid('on')
            if self.data['workdir_short'] is not None:
                ax[0].set_title(self.data['workdir_short'], fontsize=10)
            else:
                ax[0].set_title('DIII-D shot ' + str(self.data['shot']) +
                                ', ' + str(self.timeid) + ' ms')

            ax[1].plot(x_fTot, fluxTot, '-ko', lw = 2, label = 'Tot')
            ax[1].plot(x_fTot, fluxConv, '-bx', lw = 2, label = 'Conv')
            ax[1].legend(loc='best')
            ax[1].set_ylabel('$\Gamma$')
            ax[1].grid('on')
            ax[-1].set_xlabel('x')
            
            ax[0].set_xlim([np.min(x_fTot) - 0.01, np.max(x_fTot) + 0.01])
            plt.show(block = False)
                
    # ----------------------------------------------------------------------------------------
    
    def getSOLPSCarbonProfs(self, plotit = False, verbose=True, dsa = None, b2mn = None, geo = None, state = None, xport = None):
        """
        Calls b2plot to get the carbon profiles
        """
        try:
            if dsa is None:
                dsa = sut.read_dsa('dsa')
            if geo is None:
                geo = sut.read_b2fgmtry('../baserun/b2fgmtry')
            if b2mn is None:
                b2mn = sut.scrape_b2mn("b2mn.dat")                
            if state is None:
                state = sut.read_b2fstate("b2fstate")
            if xport is None:
                xport = sut.read_transport_files(".", dsa=dsa, geo=geo, state=state)

            sy = sut.avg_like_b2plot(geo['gs'][b2mn['jxa']+1,:,1])            
            nc_solps = state['na'][b2mn['jxa']+1,:,8]
            nd_solps = state['na'][b2mn['jxa']+1,:,1]
            x_nc = dsa                   
            flux_carbon = sut.avg_like_b2plot(state['fna'][b2mn['jxa']+1,:,1,8])/sy
            vr_carbon = sut.avg_like_b2plot(xport['vlay'][:,8])/sy

        except:
            print('  Falling back to b2plot to get carbon fluxes')

            if not self.b2plot_ready:
                sut.set_b2plot_dev()
                self.b2plot_ready = True

            x_nc, nc_solps = sut.B2pl("na 8 zsel writ jxa f.y")
            dummy, nd_solps = sut.B2pl("na 1 zsel writ jxa f.y")
            dummy, flux_carbon = sut.B2pl("fnay 8 zsel psy writ jxa f.y")  # x variables are the same JDL: use Z here?
            dummy, vr_carbon = sut.B2pl("vlay 8 zsel writ jxa f.y")
        
            for c in [flux_carbon, vr_carbon]:
                if not c:
                    print("WARNING: Variable not populated by b2plot in getSOLPSCarbonProfs")
                    print("  Make sure ncl_ncar and netcdf modules are loaded")
                    break

        self.data['solpsData']['profiles']['x_nC'] = np.array(x_nc)
        self.data['solpsData']['profiles']['nC'] = np.array(nc_solps)
        self.data['solpsData']['profiles']['nD'] = np.array(nd_solps)
        self.data['solpsData']['profiles']['fluxC'] = np.array(flux_carbon)
        self.data['solpsData']['profiles']['vrC'] = np.array(vr_carbon)
        
        if plotit:
            if 'psiSOLPS' in self.data['solpsData']:  # plot in psiN space if it's been calculated
                x_nc = self.data['solpsData']['psiSOLPS']
                
            f, ax = plt.subplots(3, sharex = 'all')

            ax[0].plot(x_nc, np.array(nc_solps)/1e18, '-kx', lw = 2, label = 'SOLPS')
            if 'ncprof' in self.data['expData']['fitProfs'].keys():
                nc_psi = self.data['expData']['fitPsiProf']
                nc_prof = np.array(self.data['expData']['fitProfs']['ncprof']*100)  # in  m^-3
                ax[0].plot(nc_psi, nc_prof, '-ro', lw = 2, label = 'Experiment')
                ax[0].legend(loc='best')
            elif verbose:
                print('**No carbon experimental data to plot**')
                
            ax[0].set_ylabel('n$_C$ (10$^{18}$ m$^{-3}$)')
            ax[0].grid('on')
            if self.data['workdir_short'] is not None:
                ax[0].set_title(self.data['workdir_short'], fontsize=10)
            else:
                ax[0].set_title('DIII-D shot ' + str(self.data['shot']) +
                                ', ' + str(self.timeid) + ' ms')
            
            ax[1].plot(x_nc, flux_carbon, '-kx', lw = 2, zorder = 2, label = 'Carbon flux')
            ax[1].set_ylabel('$\Gamma_C$')
            ax[1].grid('on')
            if 'fluxTot' in self.data['solpsData']['profiles']:
                ax[1].plot(x_nc, self.data['solpsData']['profiles']['fluxTot'],
                           '--bo', lw=2, zorder = 1, label = 'Total electron flux')

            ax[2].plot(x_nc, vr_carbon, '-kx', lw = 2)
            ax[2].set_ylabel('V$_{r,C}$')
            ax[2].grid('on')
            
            if 'psiSOLPS' in self.data['solpsData']:
                ax[-1].set_xlabel('$\psi_n$')
            else:
                ax[-1].set_xlabel('x')
            ax[0].set_xlim([np.min(x_nc) - 0.01, np.max(x_nc) + 0.01])
            plt.tight_layout()
            plt.show(block = False)
    
    # ----------------------------------------------------------------------------------------
    
    def calcXportCoef(self, plotit = True, Dn_min = 0.001, chie_min = 0.01, chii_min = 0.01,
                      Dn_max = 100, chie_max = 400, chii_max = 400, vrc_mag=0.0, 
                      reduce_Ti_fileloc = None, plot_gradient_method = False,
                      fractional_change = 1, exp_prof_rad_shift = 0, chii_eq_chie = False,
                      use_ratio_bc = True, debug_plots = False, verbose = False, figblock = False,
                      ti_decay_len = 0.015, te_decay_len = None, ne_decay_len = None,
                      ti_decay_min = 1, te_decay_min = 1, ne_decay_min = 1e18):
        """
        Calculates the transport coefficients to be written into b2.transport.inputfile
        
        Requires experimental profiles to have already been saved to self.data

        Inputs:
          ti_decay_len: Decay length for exponential falloff outside separatrix (units of psin)
                        (set to None to skip this)
          te_decay_len: ""
          ne_decay_len: ""
          ti_decay_min: far-SOL Ti to decay to (eV)
          te_decay_min: far-SOL Te to decay to (eV)
          ne_decay_min: far-SOL ne to decay to (m^-3)
          fractional_change: Set to number smaller than 1 if the incremental change is too large and
                             you want to take a half step or something different
          exp_prof_rad_shift: Apply a radial shift to all experimental profiles
                              (in units of psi_n, positive shifts profiles outward so separatrix is hotter)
          reduce_Ti_fileloc: Location of a saved array to get the ratio between T_C (measured) and T_i
                        Example file on GA clusters for DIII-D, calculated from Shaun Haskey's T_D measurements
                        for 171558 @ 3200 ms:
                        '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt'
          use_ratio_bc: Modifies the transport coefficients at the final grid cell proportionally to the
                        mismatch of the DC offset of the profile. Since these routines match the gradients
                        everywhere, there needs to be some way to set the scalar offset of the profile for flux BC
          vrc_mag:      Magnetude of the carbon velocity pinch
                        (shape and position are still hard coded)
        """
        # Load data that was read from SOLPS .last10 profiles

        psi_solps = self.data['solpsData']['psiSOLPS']
        dsa = self.data['solpsData']['dsa']
        neold = self.data['solpsData']['last10']['ne']
        dold = self.data['solpsData']['last10']['dn']
        teold = self.data['solpsData']['last10']['te']
        keold = self.data['solpsData']['last10']['ke']
        tiold = self.data['solpsData']['last10']['ti']
        kiold = self.data['solpsData']['last10']['ki']

        if self.data['impurities']:
            ndold = self.data['solpsData']['profiles']['nD']
        else:
            ndold = neold
        
        fluxTot = self.data['solpsData']['profiles']['fluxTot']
        fluxD = self.data['solpsData']['profiles']['fluxD']
        fluxConv = self.data['solpsData']['profiles']['fluxConv']
        qe = self.data['solpsData']['profiles']['qe']
        qi = self.data['solpsData']['profiles']['qi']
        
        psi_to_dsa_func = interpolate.interp1d(psi_solps, dsa, fill_value = 'extrapolate')

        # Convective portion of heat flux to be subtracted to get diffusive component
        # These are not actually used with the way it's coded now
        # SOLPS_qe_conv = 2.5 * dold * teold * eV
        # SOLPS_qi_conv = 2.5 * dold * tiold * eV
                
        # ne and Gamma_e

        if ne_decay_len is not None:
            print("Applying decay to ne profile")
            self.modify_ne(sol_points = 10, max_psin = np.max(psi_solps) + 0.001,
                           decay_length = ne_decay_len, rad_loc_for_exp_decay = 1.0,
                           ne_min = ne_decay_min, plotit = debug_plots)
            neexp = 1.0e20*self.data['expData']['fitProfs']['ne_mod']
            neexppsi = self.data['expData']['fitProfs']['ne_mod_psi'] + exp_prof_rad_shift
        else:
            neexp = 1.0e20 * self.data['expData']['fitProfs']['neprof']
            neexppsi = self.data['expData']['fitPsiProf'] + exp_prof_rad_shift

        dsa_neprofile = psi_to_dsa_func(neexppsi)            

        gnold = np.gradient(neold) / np.gradient(dsa)  # Only used for dnew_ratio
        gnexp = np.gradient(neexp) / np.gradient(dsa_neprofile)

        gnexp_dsafunc = interpolate.interp1d(dsa_neprofile, gnexp, kind='linear', fill_value = 'extrapolate')
        gnexp_solpslocs = gnexp_dsafunc(dsa)
        if (np.max(gnexp_solpslocs) > 0):
            print("WARNING: Positive n gradient found at dsa =",dsa[np.argmax(gnexp_solpslocs)])
            print("         Modify fits or min Dn value will be used here")
        
        # psi_to_dsa_func function only valid in SOLPS range,
        # so gnexp_dsafunc shouldn't be applied outside that
        gnexp_solpslocs_dsa = gnexp_dsafunc(dsa)

        # Set boundary condition to get ne[-1] right
        expden_dsa_func = interpolate.interp1d(dsa_neprofile, neexp, fill_value = 'extrapolate')

        ne_decay_len_end = (expden_dsa_func(dsa[-2]) - expden_dsa_func(dsa[-1])) / \
            np.mean([expden_dsa_func(dsa[-1]), expden_dsa_func(dsa[-2])])

        if verbose: print('ne_decay_len = ' + str(ne_decay_len))
        gnexp_solpslocs[-1] = -expden_dsa_func(dsa[-1]) / ne_decay_len_end

        # this method assumes no convective transport (ok in some cases)
        dnew_ratio = (gnold / gnexp_solpslocs_dsa) * dold

        flux = fluxD - fluxConv  # Conductive portion of the total flux
        dnew_flux = -flux / gnexp_solpslocs

        if use_ratio_bc:
            dnew_ratio[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
            dnew_flux[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
        

        dnew_ratio[0] = dnew_ratio[1] # guard cells
        dnew_flux[0] = dnew_flux[1]
        
        # Te and ke
        if te_decay_len is not None:
            print("Applying decay to Te profile")            
            self.modify_te(sol_points = 10, max_psin = np.max(psi_solps) + 0.001,
                           decay_length = te_decay_len, rad_loc_for_exp_decay = 1.0,
                           te_min = te_decay_min, plotit = debug_plots)

            teexp = 1.0e3*self.data['expData']['fitProfs']['te_mod']
            teexppsi = self.data['expData']['fitProfs']['te_mod_psi'] + exp_prof_rad_shift
        else:
            teexp = 1.0e3 * self.data['expData']['fitProfs']['teprof']
            teexppsi = self.data['expData']['fitPsiProf'] + exp_prof_rad_shift

        dsa_teprofile = psi_to_dsa_func(teexppsi)
        
        gteold = np.gradient(teold) / np.gradient(dsa)
        gteexp = np.gradient(teexp) / np.gradient(dsa_teprofile)

        gteexp_dsafunc = interpolate.interp1d(dsa_teprofile, gteexp, kind='linear', fill_value = 'extrapolate')
        gteexp_solpslocs = gteexp_dsafunc(dsa)
        if (np.max(gteexp_solpslocs) > 0):
            print("WARNING: Positive Te gradient found at dsa =",dsa[np.argmax(gteexp_solpslocs)])
            print("         Modify fits or min chie value will be used here")
                
        # Set boundary condition to get Te[-1] right
        expTe_dsa_func = interpolate.interp1d(dsa_teprofile, teexp, fill_value = 'extrapolate')
        te_decay_len_end = (expTe_dsa_func(dsa[-2]) - expTe_dsa_func(dsa[-1])) / \
            np.mean([expTe_dsa_func(dsa[-1]), expTe_dsa_func(dsa[-2])])
            
        if verbose: print('Te_decay_len = ' + str(te_decay_len))
        gteexp_solpslocs[-1] = -expTe_dsa_func(dsa[-1]) / te_decay_len_end
        
        kenew_ratio = (gteold / gteexp_solpslocs) * keold

        # gradient has to be in dsa to work
        kenew_flux = -(qe - 2.5 * fluxTot * teold * eV) / (neold * eV * gteexp_solpslocs)

        if use_ratio_bc:
            kenew_ratio[-1] = keold[-1] * teold[-1] / expTe_dsa_func(dsa[-1])
            kenew_flux[-1] = keold[-1] * teold[-1] / expTe_dsa_func(dsa[-1])

        kenew_ratio[0] = kenew_ratio[1]   # guard cells
        kenew_flux[0] = kenew_flux[1]
        
        # Ti and ki

        if reduce_Ti_fileloc or (ti_decay_len is not None):
            self.modify_ti(ratio_fileloc = reduce_Ti_fileloc, sol_points = 10, max_psin = np.max(psi_solps) + 0.001,
                           decay_length = ti_decay_len, rad_loc_for_exp_decay = 1.0,
                           plotit = debug_plots, reduce_ti = (reduce_Ti_fileloc is not None), ti_min = ti_decay_min)

            tiexp = 1.0e3*self.data['expData']['fitProfs']['ti_mod']
            tiexppsi = self.data['expData']['fitProfs']['ti_mod_psi'] + exp_prof_rad_shift

        else:
            tiexp = 1.0e3*self.data['expData']['fitProfs']['tiprof']
            tiexppsi = self.data['expData']['fitProfs']['tipsi'] + exp_prof_rad_shift
        
        dsa_tiprofile = psi_to_dsa_func(tiexppsi)
        
        gtiold = np.gradient(tiold) / np.gradient(dsa)
        gtiexp = np.gradient(tiexp) / np.gradient(dsa_tiprofile)
        
        gtiexp_dsafunc = interpolate.interp1d(dsa_tiprofile, gtiexp, kind='linear', fill_value = 'extrapolate')
        gtiexp_solpslocs = gtiexp_dsafunc(dsa)
        if (np.max(gtiexp_solpslocs) > 0):
            print("WARNING: Positive Ti gradient found at dsa =",dsa[np.argmax(gtiexp_solpslocs)])
            print("         Modify fits or min chii value will be used here")
            
        # Set boundary condition to get Ti[-1] right
        expTi_dsa_func = interpolate.interp1d(dsa_tiprofile, tiexp, fill_value = 'extrapolate')
        if ti_decay_len is not None:
            gtiexp_solpslocs[-1] = -expTi_dsa_func(dsa[-1]) / ti_decay_len    
        
        kinew_ratio = (gtiold / gtiexp_solpslocs) * kiold

        # gradient has to be in dsa to work
        kinew_flux = -(qi - 2.5 * fluxTot * tiold * eV) / (ndold * eV * gtiexp_solpslocs)

        if use_ratio_bc:
            kinew_ratio[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])
            kinew_flux[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])

        kinew_ratio[0] = kinew_ratio[1]   # guard cells
        kinew_flux[0] = kinew_flux[1]

        if fractional_change != 1:
            dnew_ratio = dnew_ratio * fractional_change + dold * (1 - fractional_change)
            dnew_flux = dnew_flux * fractional_change + dold * (1 - fractional_change)
            kenew_ratio = kenew_ratio * fractional_change + keold * (1 - fractional_change)
            kenew_flux = kenew_flux * fractional_change + keold * (1 - fractional_change)
            kinew_ratio = kinew_ratio * fractional_change + kiold * (1 - fractional_change)
            kinew_flux = kinew_flux * fractional_change + kiold * (1 - fractional_change)

        # Apply constraints
        
        dnew_ratio[dnew_ratio < Dn_min] = Dn_min
        dnew_flux[dnew_flux < Dn_min] = Dn_min
        kinew_ratio[kinew_ratio < chii_min] = chii_min
        kinew_flux[kinew_flux < chii_min] = chii_min
        kenew_ratio[kenew_ratio < chie_min] = chie_min
        kenew_flux[kenew_flux < chie_min] = chie_min
        
        dnew_ratio[dnew_ratio > Dn_max] = Dn_max
        dnew_flux[dnew_flux > Dn_max] = Dn_max
        kinew_ratio[kinew_ratio > chii_max] = chii_max
        kinew_flux[kinew_flux > chii_max] = chii_max
        kenew_ratio[kenew_ratio > chie_max] = chie_max
        kenew_flux[kenew_flux > chie_max] = chie_max
        
        
        # Carbon transport coefficients

        if 'c' in self.data['impurities']:
            # vrc_mag = 20.0  # 60 for gauss
            vr_pos = 0.97
            # vr_wid = 0.02
            # sig = vr_wid / 2.3548
            # vr_shape = np.exp(-((psi_solps - vr_pos) ** 2) / (2 * sig ** 2))  # Gaussian
            
            vr_shape = psi_solps - vr_pos  # linear
            
            vr_carbon = vr_shape * vrc_mag / max(vr_shape)

            # default to C diffusion being the same as main ion
            # D_carbon = Dn_min + 0.0 * dnew_flux
            # D_carbon[19:] = dnew_flux[19:]
            # D_carbon[1:] = dnew_flux[1:]
            D_carbon = None
                    
        else:
            vr_carbon = None
            D_carbon = None

        coef_limits = {'Dn_min':Dn_min, 'Dn_max':Dn_max, 'chie_min':chie_min,
                       'chii_min':chii_min, 'chie_max':chie_max, 'chii_max':chii_max}


        self.data['solpsData']['xportCoef'] = {'dnew_ratio': dnew_ratio, 'dnew_flux': dnew_flux,
                                               'kenew_ratio': kenew_ratio, 'kenew_flux':kenew_flux,
                                               'kinew_ratio': kinew_ratio, 'kinew_flux':kinew_flux,
                                               'vr_carbon': vr_carbon, 'D_carbon': D_carbon,
                                               'limits': coef_limits, 'exp_prof_shift': exp_prof_rad_shift}
        if plotit:
            self.plotXportCoef(figblock=figblock, plot_Ti = not chii_eq_chie,
                               plot_older=('ne_old' in self.data['solpsData']['last10'].keys()),
                               include_gradient_method=plot_gradient_method)

        if debug_plots:
            plt.figure()
            plt.plot(psi_solps, gtiexp_solpslocs / 1e3, label='gtiexp_solpslocs')
            plt.plot(psi_solps, gtiold / 1e3, label='gtiold')
            plt.plot(psi_solps, gteexp_solpslocs / 1e3, '--', lw=2, label='gteexp_solpslocs')
            plt.plot(psi_solps, gteold / 1e3, '--', lw=2, label='gteold')
            plt.xlabel('$\psi_N$')
            plt.ylabel('$\\nabla$T / $\\nabla\psi_N$')
            plt.legend(loc='best')

            plt.figure()
            plt.plot(tiexppsi, 1.0e3 * self.data['expData']['fitProfs']['tiprof'],
                     'k', lw=2, label='Ti measured')
            plt.plot(tiexppsi, tiexp, 'r', label='Ti_new')
            plt.xlabel('$\psi_N$')
            plt.ylabel('T$_i$ (keV)')
            plt.legend(loc='best')

            if 'c' in self.data['impurities']:
                plt.figure()
                plt.plot(psi_solps, dnew_ratio, '-or', label='dnew_ratio')
                plt.plot(psi_solps, dnew_flux, '-ob', label='dnew_flux')
                plt.plot(psi_solps, D_carbon, '-xg', label='D_carbon')
                plt.legend(loc='best')

            plt.show(block=figblock)

    # ----------------------------------------------------------------------------------------

    def flatSOLcoeffs(self, prof_choice, psin_start, coef_val, plotit=False, figblock = False):
        """
        Manually change transport coefficients outside of some position to a specified value
        Useful if profile flattens (e.g., a density shoulder) and a precise match is difficult or unnecessary

        Inputs:
          prof_choice    Choose from 'd' (particle diffusivity), 'ke' (electron thermal diffusivity),
                         or 'ki' (ion thermal diffusivity)
          psin_start     Starting position to set the flat coefficients (goes outward from here)
        """
        if prof_choice.lower() not in ['d', 'ke', 'ki']:
            print('WARNING: invalid choice for diffusion coefficient to set in SOL')
            print('  you selected: ' + prof_choice)
            print('  available options: "d" (particle diffusivity), "ke" (electron thermal diffusivity),' +
                  ' or "ki" (ion thermal diffusivity)')
            return

        psi_solps = self.data['solpsData']['psiSOLPS']
        ratio_key = prof_choice.lower() + 'new_ratio'
        flux_key = prof_choice.lower() + 'new_flux'

        self.data['solpsData']['xportCoef'][ratio_key][psi_solps > psin_start] = coef_val
        self.data['solpsData']['xportCoef'][flux_key][psi_solps > psin_start] = coef_val

        if plotit:
            self.plotXportCoef(plot_Ti=(prof_choice.lower() == 'ki'), figblock=figblock)

    # ----------------------------------------------------------------------------------------

    def plotXportCoef(self, figblock=False, figsize=(14,7), plot_Ti = True, plot_older = False,
                      include_gradient_method = False):
        """
        Plot the upstream profiles from SOLPS compared to the experiment
        along with the corresponding updated transport coefficients
        """
        if include_gradient_method:
            updated_flux_label = 'Updated (fluxes)'
        else:
            updated_flux_label = 'Updated'

        dnew_ratio = self.data['solpsData']['xportCoef']['dnew_ratio']
        dnew_flux = self.data['solpsData']['xportCoef']['dnew_flux']
        kenew_ratio = self.data['solpsData']['xportCoef']['kenew_ratio']
        kenew_flux = self.data['solpsData']['xportCoef']['kenew_flux']
        kinew_ratio = self.data['solpsData']['xportCoef']['kinew_ratio']
        kinew_flux = self.data['solpsData']['xportCoef']['kinew_flux']
        coef_limits = self.data['solpsData']['xportCoef']['limits']

        exp_prof_shift = self.data['solpsData']['xportCoef']['exp_prof_shift']
        
        if 'te_mod' in self.data['expData']['fitProfs'].keys():            
            teexp = 1.0e3*self.data['expData']['fitProfs']['te_mod']
            teexppsi = self.data['expData']['fitProfs']['te_mod_psi'] + exp_prof_shift
        else:
            teexp = 1.0e3 * self.data['expData']['fitProfs']['teprof']
            teexppsi = self.data['expData']['fitPsiProf'] + exp_prof_shift

        if 'ne_mod' in self.data['expData']['fitProfs'].keys():            
            neexp = 1.0e20*self.data['expData']['fitProfs']['ne_mod']
            neexppsi = self.data['expData']['fitProfs']['ne_mod_psi'] + exp_prof_shift
        else:
            neexp = 1.0e20*self.data['expData']['fitProfs']['neprof']      
            neexppsi = self.data['expData']['fitPsiProf'] + exp_prof_shift

        if 'ti_mod' in self.data['expData']['fitProfs'].keys():
            tiexp = 1.0e3*self.data['expData']['fitProfs']['ti_mod']
            tiexppsi = self.data['expData']['fitProfs']['ti_mod_psi'] + exp_prof_shift
        else:
            tiexp = 1.0e3*self.data['expData']['fitProfs']['tiprof']            
            tiexppsi = self.data['expData']['fitProfs']['tipsi'] + exp_prof_shift

        psi_solps = self.data['solpsData']['psiSOLPS']
        neold = self.data['solpsData']['last10']['ne']
        dold = self.data['solpsData']['last10']['dn']
        teold = self.data['solpsData']['last10']['te']
        keold = self.data['solpsData']['last10']['ke']
        tiold = self.data['solpsData']['last10']['ti']
        kiold = self.data['solpsData']['last10']['ki']

        # Check if last10.old profiles exist
        if plot_older and ('ne_old' in self.data['solpsData']['last10'].keys()):
            neolder = self.data['solpsData']['last10']['ne_old']
            dolder = self.data['solpsData']['last10']['dn_old']
            teolder = self.data['solpsData']['last10']['te_old']
            keolder = self.data['solpsData']['last10']['ke_old']
            tiolder = self.data['solpsData']['last10']['ti_old']
            kiolder = self.data['solpsData']['last10']['ki_old']
        else:
            plot_older = False

        # Find limits for plots
        Te_inds_in_range = np.where(teexppsi > np.min(psi_solps))[0]
        ne_inds_in_range = np.where(neexppsi > np.min(psi_solps))[0]        
        Ti_inds_in_range = np.where(tiexppsi > np.min(psi_solps))[0]
        max_Te = np.max([np.max(teold), np.max(teexp[Te_inds_in_range])]) / 1.0e3
        max_Ti = np.max([np.max(tiold), np.max(tiexp[Ti_inds_in_range])]) / 1.0e3
        max_ne = np.max([np.max(neold), np.max(neexp[ne_inds_in_range])]) / 1.0e19
        max_dn = np.max([np.max(dold), np.max(dnew_ratio), np.max(dnew_flux)])
        max_ke = np.max([np.max(keold), np.max(kenew_ratio), np.max(kenew_flux)])
        max_ki = np.max([np.max(kiold), np.max(kinew_ratio), np.max(kinew_flux)])
        min_dn = np.min([np.min(dold), np.min(dnew_ratio), np.min(dnew_flux)])
        min_ke = np.min([np.min(keold), np.min(kenew_ratio), np.min(kenew_flux)])
        min_ki = np.min([np.min(kiold), np.min(kinew_ratio), np.min(kinew_flux)])


        headroom = 1.05
        xlims = [np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01]

        if plot_Ti:
            nplots = 3
        else:
            nplots = 2

        f, ax = plt.subplots(2, nplots, sharex = 'all', figsize=figsize)
        if plot_older:
            ax[0, 0].plot(psi_solps, neolder / 1.0e19, '--g', lw = 1, label = 'previous SOLPS')
        ax[0, 0].plot(neexppsi, neexp / 1.0e19, '--bo', lw = 1, label = 'Exp. data')
        ax[0, 0].plot(psi_solps, neold / 1.0e19, '-xr', lw = 2, label = 'SOLPS')
        ax[0, 0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0, 0].legend(loc = 'best', fontsize=12)
        ax[0, 0].set_ylim([0, max_ne*headroom])
        ax[0, 0].grid('on')

        if coef_limits['Dn_min'] is not None:
            ax[1, 0].semilogy(xlims, [coef_limits['Dn_min'], coef_limits['Dn_min']], '--m')
        if coef_limits['Dn_max'] is not None:
            ax[1, 0].semilogy(xlims, [coef_limits['Dn_max'], coef_limits['Dn_max']], '--m')
        if plot_older:
            ax[1, 0].semilogy(psi_solps, dolder, '--g', lw = 1, label = 'previous iteration')
        ax[1, 0].semilogy(psi_solps, dold, '-xr', lw = 2, label = 'SOLPS input')
        ax[1, 0].semilogy(psi_solps, dnew_flux, '-ok', lw = 2, label = updated_flux_label)
        if include_gradient_method:
            ax[1, 0].semilogy(psi_solps, dnew_ratio, '-+c', lw = 1, label = 'Updated (gradients)')
        ax[1, 0].set_ylabel('D (m$^2$/s)')
        ax[1, 0].set_xlabel('$\psi_N$')
        ax[1, 0].set_ylim([min_dn/np.sqrt(headroom), max_dn*headroom])
        ax[1, 0].grid('on')
        ax[0, 1].plot(teexppsi, teexp / 1.0e3, '--bo', lw = 1, label = 'Exp. Data')

        if plot_older:
            ax[0, 1].plot(psi_solps, teolder / 1.0e3, '--g', lw = 1, label = 'SOLPS old')
        ax[0, 1].plot(psi_solps, teold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 1].set_ylabel('T$_e$ (keV)')
        ax[0, 1].set_ylim([0, max_Te*headroom])
        ax[0, 1].grid('on')
        if self.data['workdir_short'] is not None:
            ax[0, 1].set_title(self.data['workdir_short'], fontsize=10)

        if coef_limits['chie_min'] is not None:
            ax[1, 1].semilogy(xlims, [coef_limits['chie_min'], coef_limits['chie_min']], '--m')
        if coef_limits['chie_max'] is not None:
            ax[1, 1].semilogy(xlims, [coef_limits['chie_max'], coef_limits['chie_max']], '--m')
        if plot_older:
            ax[1, 1].semilogy(psi_solps, keolder, '--g', lw = 1, label = 'previous iteration')
        ax[1, 1].semilogy(psi_solps, keold, '-xr', lw = 2, label = 'SOLPS input')
        ax[1, 1].semilogy(psi_solps, kenew_flux, '-ok', lw = 2, label = updated_flux_label)
        if include_gradient_method:
            ax[1, 1].semilogy(psi_solps, kenew_ratio, '-+c', lw = 1, label = 'Updated (gradients)')
        ax[1, 1].set_ylabel('$\chi_e$ (m$^2$/s)')
        ax[1, 1].set_xlabel('$\psi_N$')
        ax[1, 1].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        ax[1, 1].set_ylim([min_ke/np.sqrt(headroom), max_ke*headroom])
        ax[1, 1].grid('on')

        if plot_Ti:
            ax[0, 2].plot(tiexppsi, tiexp / 1.0e3, '--bo', lw = 1, label = 'Exp. Data')
            if plot_older:
                ax[0, 2].plot(psi_solps, tiolder / 1.0e3, '--g', lw = 1, label = 'SOLPS old')
            ax[0, 2].plot(psi_solps, tiold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
            ax[0, 2].set_ylabel('T$_i$ (keV)')
            ax[0, 2].set_ylim([0, max_Ti*headroom])
            ax[0, 2].grid('on')

            if coef_limits['chii_min'] is not None:
                ax[1, 2].semilogy(xlims, [coef_limits['chii_min'], coef_limits['chii_min']], '--m')
            if coef_limits['chii_max'] is not None:
                ax[1, 2].semilogy(xlims, [coef_limits['chii_max'], coef_limits['chii_max']], '--m')
            if plot_older:
                ax[1, 2].semilogy(psi_solps, kiolder, '--g', lw = 1, label = 'previous iteration')
            ax[1, 2].semilogy(psi_solps, kiold, '-xr', lw = 2, label = 'SOLPS input')
            ax[1, 2].semilogy(psi_solps, kinew_flux, '-ok', lw = 2, label = updated_flux_label)
            if include_gradient_method:
                ax[1, 2].semilogy(psi_solps, kinew_ratio, '-+c', lw = 1, label = 'Updated (gradients)')
            ax[1, 2].set_ylabel('$\chi_i$ (m$^2$/s)')
            ax[1, 2].set_xlabel('$\psi_N$')
            ax[1, 2].set_xlim(xlims)
            ax[1, 2].set_ylim([min_ki/np.sqrt(headroom), max_ki*headroom])
            ax[1, 2].grid('on')

        ax[1, -1].legend(loc='best', fontsize=10)
        if xlims[0] > 0.8:
            ax[0, 0].set_xticks(np.arange(0.84, 1.05, 0.04))
        ax[0, 0].set_xlim(xlims)
        plt.tight_layout()

        plt.show(block=figblock)

    # ----------------------------------------------------------------------------------------
    
    def plot_profiles(self, include_ti = True, headroom = 1.05):
        """
        Plot the upstream profiles from SOLPS compared to the experiment
        """

        # Load SOLPS profiles and transport coefficients

        psi_solps = self.data['solpsData']['psiSOLPS']
        nesolps = self.data['solpsData']['last10']['ne']
        tesolps = self.data['solpsData']['last10']['te'] * 1.0e-3
        tisolps = self.data['solpsData']['last10']['ti'] * 1.0e-3
        
        # Load experimental profiles

        psi_TSfit = self.data['expData']['fitPsiProf'] + self.data['solpsData']['xportCoef']['exp_prof_shift']
        nefit = 1.0e20 * self.data['expData']['fitProfs']['neprof']
        tefit = self.data['expData']['fitProfs']['teprof']
        tifit = self.data['expData']['fitProfs']['tiprof']
        tifitpsi = self.data['expData']['fitProfs']['tipsi'] + self.data['solpsData']['xportCoef']['exp_prof_shift']

        rawdat_keys = ['nedatpsi', 'tedatpsi']
        rawdat_scalars = [10, 1.0]  # ne saved as 10^20, we want 10^19
        if include_ti:
            rawdat_keys.append('tidatpsi')
            rawdat_scalars.append(1.0)
        nprofs = len(rawdat_keys)


        # Find limits of Te, Ti for plots
        Te_inds_in_range = np.where(psi_TSfit > np.min(psi_solps))[0]
        ne_inds_in_range = np.where(psi_TSfit > np.min(psi_solps))[0]
        Ti_inds_in_range = np.where(tifitpsi > np.min(psi_solps))[0]
        max_ne = np.max([np.max(nesolps), np.max(nefit[ne_inds_in_range])]) / 1.0e19
        max_Te = np.max([np.max(tesolps), np.max(tefit[Te_inds_in_range])])
        max_Ti = np.max([np.max(tisolps), np.max(tifit[Ti_inds_in_range])])

        f, ax = plt.subplots(nprofs, sharex = 'all')

        for i in range(nprofs):
            ax[i].errorbar(self.data['expData']['fitVals'][rawdat_keys[i]]['x'],
                           self.data['expData']['fitVals'][rawdat_keys[i]]['y'] * rawdat_scalars[i],
                           self.data['expData']['fitVals'][rawdat_keys[i]]['yerr'] * rawdat_scalars[i],
                           xerr=None, fmt='o', ls='', c='k', mfc='None', mec='k',
                           zorder=1, label='Experimental Data')

        if 'ne_mod' in self.data['expData']['fitProfs'].keys():
            ax[0].plot(self.data['expData']['fitProfs']['ne_mod_psi'],
                       self.data['expData']['fitProfs']['ne_mod'],
                       '--k', lw=2, zorder=3, label = 'Experimental Fit')            
        else:
            ax[0].plot(psi_TSfit, nefit / 1.0e19, '--k', lw=2, zorder=3, label='Experimental Fit')
        # ax[0].plot(psi_solps, nesolps / 1.0e19, 'xr', lw=2, mew=2, ms=10, label='SOLPS')
        ax[0].plot(psi_solps, nesolps / 1.0e19, '-r', lw=2, zorder=2, label='SOLPS')
        ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0].legend(loc='best', fontsize=14)
        ax[0].set_ylim([0, max_ne * headroom])
        if 'te_mod' in self.data['expData']['fitProfs'].keys():
            ax[1].plot(self.data['expData']['fitProfs']['te_mod_psi'],
                       self.data['expData']['fitProfs']['te_mod'],
                       '--k', lw=2, zorder=3, label = 'Experimental Fit')
        else:
            ax[1].plot(psi_TSfit, tefit, '--k', lw=2, zorder=3, label='Experimental Fit')
        # ax[1].plot(psi_solps, tesolps, 'xr', mew=2, ms=10, label='SOLPS')
        ax[1].plot(psi_solps, tesolps, '-r', lw=2, zorder=2, label='SOLPS')
        ax[1].set_ylabel('T$_e$ (keV)')
        ax[1].set_ylim([0, max_Te * headroom])
        ax[1].set_yticks(np.arange(0, max_Te * headroom + 0.2, 0.2))

        if include_ti:
            # ax[2].plot(psi_solps, tisolps, 'xr', mew = 2, ms = 10, label = 'SOLPS')
            ax[2].plot(tifitpsi, tifit, '--k', lw = 2, zorder=3, label = 'Experimental Fit')
            ax[2].plot(psi_solps, tisolps, '-r', lw = 2, zorder=2, label = 'SOLPS')

            if 'ti_mod' in self.data['expData']['fitProfs'].keys():
                ax[2].plot(self.data['expData']['fitProfs']['ti_mod_psi'],
                           self.data['expData']['fitProfs']['ti_mod'],
                           '--b', lw=2, zorder=4, label = 'Modified Ti fit')
            ax[2].set_ylabel('T$_i$ (keV)')
            ax[2].set_ylim([0, max_Ti * headroom])
            ax[2].set_yticks(np.arange(0, max_Ti * headroom + 0.2, 0.2))

        for i in range(nprofs):
            ax[i].grid('on')

        ax[0].set_xticks(np.arange(0.84, 1.05, 0.04))
        ax[0].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.004])
        ax[-1].set_xlabel('$\psi_N$')
        ax[0].set_title('Upstream profiles', fontsize=16)
        plt.tight_layout()

        plt.show(block = False)

    # ----------------------------------------------------------------------------------------

    def plot_matching_case(self, include_ti=True, headroom=1.02):
        """
        Plot the upstream profiles from SOLPS compared to the experiment, as well as diffusion coefficients
        """
        if 'xportCoef' not in self.data['solpsData']:
            print('Transport coefficients not yet calculated!! Calculating them using defaults')
            self.calcXportCoef(plotit = False, debug_plots = False)

        # Load SOLPS profiles and transport coefficients

        psi_solps = self.data['solpsData']['psiSOLPS']
        nesolps = self.data['solpsData']['last10']['ne']
        dsolps = self.data['solpsData']['last10']['dn']
        tesolps = self.data['solpsData']['last10']['te'] * 1.0e-3
        kesolps = self.data['solpsData']['last10']['ke']
        tisolps = self.data['solpsData']['last10']['ti'] * 1.0e-3
        kisolps = self.data['solpsData']['last10']['ki']

        # Load experimental profiles

        psi_data_fit = self.data['expData']['fitPsiProf'] # JDL need shift here?
        nefit = 1.0e20 * self.data['expData']['fitProfs']['neprof']
        tefit = self.data['expData']['fitProfs']['teprof']
        tifit = self.data['expData']['fitProfs']['tiprof']
        tifitpsi = self.data['expData']['fitProfs']['tipsi']

        rawdat_keys = ['nedatpsi', 'tedatpsi']
        rawdat_scalars = [10, 1.0]  # ne saved as 10^20, we want 10^19
        if include_ti:
            rawdat_keys.append('tidatpsi')
            rawdat_scalars.append(1.0)
        nprofs = len(rawdat_keys)

        # Find limits of Te, Ti for plots
        Te_inds_in_range = np.where((self.data['expData']['fitVals']['tedatpsi']['x'] > np.min(psi_solps)) &
                                    (self.data['expData']['fitVals']['tedatpsi']['x'] < np.max(psi_solps)))[0]
        ne_inds_in_range = np.where((self.data['expData']['fitVals']['nedatpsi']['x'] > np.min(psi_solps)) &
                                    (self.data['expData']['fitVals']['nedatpsi']['x'] < np.max(psi_solps)))[0]
        Ti_inds_in_range = np.where((self.data['expData']['fitVals']['tidatpsi']['x'] > np.min(psi_solps)) &
                                    (self.data['expData']['fitVals']['tidatpsi']['x'] < np.max(psi_solps)))[0]
        max_raw_ne = np.max(self.data['expData']['fitVals']['nedatpsi']['y'][ne_inds_in_range] +
                            self.data['expData']['fitVals']['nedatpsi']['yerr'][ne_inds_in_range])
        max_raw_te = np.max(self.data['expData']['fitVals']['tedatpsi']['y'][Te_inds_in_range] +
                            self.data['expData']['fitVals']['tedatpsi']['yerr'][Te_inds_in_range])
        max_raw_ti = np.max(self.data['expData']['fitVals']['tidatpsi']['y'][Ti_inds_in_range] +
                            self.data['expData']['fitVals']['tidatpsi']['yerr'][Ti_inds_in_range])
        max_ne = np.max([np.max(nesolps) / 1.0e19, max_raw_ne *10])
        max_Te = np.max([np.max(tesolps), max_raw_te])
        max_Ti = np.max([np.max(tisolps), max_raw_ti])
        max_temp = np.max([max_Te, max_Ti])  # just use this so they're on the same scale

        f, ax = plt.subplots(2, nprofs, sharex='all')

        for i in range(nprofs):
            ax[0, i].errorbar(self.data['expData']['fitVals'][rawdat_keys[i]]['x'],
                              self.data['expData']['fitVals'][rawdat_keys[i]]['y'] * rawdat_scalars[i],
                              self.data['expData']['fitVals'][rawdat_keys[i]]['yerr'] * rawdat_scalars[i],
                              xerr=None, fmt='o', ls='', c='k', mfc='None', mec='k',
                              zorder=1, label='Experimental Data')

        ax[0, 0].plot(psi_data_fit, nefit / 1.0e19, '--k', lw=2, zorder=3, label='Experimental Fit')
        ax[0, 0].plot(psi_solps, nesolps / 1.0e19, '-r', lw=2, zorder=2, label='SOLPS')
        ax[0, 0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0, 0].legend(loc='best', fontsize=14)
        ax[0, 0].set_ylim([0, max_ne * headroom])

        ax[1, 0].semilogy(psi_solps, dsolps, '-r', lw = 2)
        ax[1, 0].set_ylabel('D')

        ax[0, 1].plot(psi_data_fit, tefit, '--k', lw=2, zorder=3, label='Experimental Fit')
        ax[0, 1].plot(psi_solps, tesolps, '-r', lw=2, zorder=2, label='SOLPS')
        ax[0, 1].set_ylabel('T$_e$ (keV)')
        ax[0, 1].set_ylim([0, max_temp * headroom])
        ax[0, 1].set_yticks(np.arange(0, max_temp * headroom + 0.2, 0.2))

        ax[1, 1].semilogy(psi_solps, kesolps, '-r', lw = 2)
        ax[1, 1].set_ylabel('$\chi_e$')

        if include_ti:
            # ax[2].plot(psi_solps, tisolps, 'xr', mew = 2, ms = 10, label = 'SOLPS')
            ax[0, 2].plot(psi_solps, tisolps, '-r', lw=2, zorder=2, label = 'SOLPS')
            ax[0, 2].plot(tifitpsi, tifit, '--k', lw=2, zorder=3, label='Experimental Fit')

            if 'ti_mod' in self.data['expData']['fitProfs'].keys():
                ax[0, 2].plot(self.data['expData']['fitProfs']['ti_mod_psi'],
                              self.data['expData']['fitProfs']['ti_mod'],
                              '--b', lw=2, zorder=4, label='Modified Ti fit')
            ax[0, 2].set_ylabel('T$_i$ (keV)')
            ax[0, 2].set_ylim([0, max_temp * headroom])
            ax[0, 2].set_yticks(np.arange(0, max_temp * headroom + 0.2, 0.2))

            ax[1, 2].semilogy(psi_solps, kisolps, '-r', lw = 2)
            ax[1, 2].set_ylabel('$\chi_i$')

        for i in range(nprofs):
            ax[1, i].set_xlabel('$\psi_N$')
            for j in range(2):
                ax[j, i].grid('on')

        ax[0, 0].set_xticks(np.arange(0.84, 1.05, 0.04))
        ax[0, 0].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.004])
        plt.tight_layout()

        plt.show(block=False)

    # ----------------------------------------------------------------------------------------
    
    def writeXport(self, new_filename = 'b2.transport.inputfile_new', fractional_change = 1, solps5_0 = False,
                   scale_D = 1, chie_use_grad = False, chii_use_grad = False, chii_eq_chie = False):
        """
        Write the b2.transport.inputfile using values saved in this object
        SOLPS5.0 was deprecated, need to modify this code if you want to write to runs that old

        Inputs:
          fractional_change  Set to number smaller than 1 if the incremental change is too large and
                             you want to take a half step or something different
          ke/i_use_grad      Use ratio of the gradients for new values of chi_i/e
          scale_D            Scalar factor to modify all particle diffusion coefficients
                             (when going from density BC to flux BC, need to reduce the transport by a
                             factor proportional to the difference in core flux between the two cases)
          chii_eq_chie       Set chi_i = chi_e, if ion data is bad or non-existent (default = False)
        """
        # dictionary defining number of electrons for possible impurity species
        n_electrons = {'he':2, 'li':3, 'be':4, 'b':5, 'c':6, 'n':7, 'ne':10, 'ar':18, 'kr':36, 'w':74}

        species_nums = [1]  # indeces of all plasma species (0 is fluid neutral D, 1 is D+)
        next_species = 3  # Exclude fluid neutrals (0, 2, 9, ...)

        for imp in self.data['impurities']:  # append list of plasma species indeces with all included impurities
            species_nums = species_nums + list(range(next_species, next_species + n_electrons[imp]))
            next_species += n_electrons[imp] + 1  # exclude fluid neutrals

        wdir = self.data['workdir']
        inFile = os.path.join(wdir, new_filename)
        if os.path.isfile(inFile):
            print("'" + new_filename + "' already exists, renaming existing " +
                  "file to 'old_xport_coef' and writing new one")
            movedfile = os.path.join(wdir, 'old_xport_coef')
            cmds = 'cp ' + inFile + ' ' + movedfile
            os.system(cmds)
        
        rn = self.data['solpsData']['last10']['rx']
        dn = self.data['solpsData']['xportCoef']['dnew_flux'] * scale_D
        if chie_use_grad:
            ke = self.data['solpsData']['xportCoef']['kenew_ratio']
        else:
            ke = self.data['solpsData']['xportCoef']['kenew_flux']

        if chii_eq_chie:
            ki = ke
        else:
            if chii_use_grad:
                ki = self.data['solpsData']['xportCoef']['kinew_ratio']
            else:
                ki = self.data['solpsData']['xportCoef']['kinew_flux']

        vrc = self.data['solpsData']['xportCoef']['vr_carbon']

        # Default is that dc = dn
        if self.data['solpsData']['xportCoef']['D_carbon'] is None:
            dc = dn
        else:
            dc = self.data['solpsData']['xportCoef']['D_carbon'] * scale_D
        
        # Step the boundary points out a tiny bit so that they are
        # interpolated onto the SOLPS grid correctly
        delta_step = 0.1*np.min(np.abs(np.diff(rn)))
        
        # Remove any small negative diffusivities and throw a warning
        
        for i in range(len(rn)):
            if dn[i] < 0:
                print('Negative diffusivity calculated! Modifying to a small positive number')
                print('dn[{}] = {:e}'.format(i,dn[i]))
                print('  Changed to dn[{}] = {:e}'.format(i,-dn[i]*1e-5))
                dn[i] = -dn[i] * 1e-5
            if ke[i] < 0:
                print('Negative thermal diffusivity calculated! Modifying to a small positive number')
                print('ke[{}] = {:e}'.format(i,ke[i]))
                print('  Changed to ke[{}] = {:e}'.format(i,-ke[i]*1e-2))
                ke[i] = -ke[i] * 1e-2
            if ki[i] < 0:
                print('Negative thermal diffusivity calculated! Modifying to a small positive number')
                print('ki[{}] = {:e}'.format(i,ki[i]))
                print('  Changed to ki[{}] = {:e}'.format(i,-ki[i]*1e-2))
                ki[i] = -ki[i] * 1e-2

        # Take a smaller step if requested

        if fractional_change != 1:
            dold = self.data['solpsData']['last10']['dn']
            keold = self.data['solpsData']['last10']['ke']
            kiold = self.data['solpsData']['last10']['ki']

            dn = dn * fractional_change + dold * (1 - fractional_change)
            ke = ke * fractional_change + keold * (1 - fractional_change)
            ki = ki * fractional_change + kiold * (1 - fractional_change)
        
        inlines = list()
        
        if solps5_0:
            print('WARNING: SOLPS5.0 is not supported with this version of the code')

            """
            inlines.append('&TRANSPORT\n')
            inlines.append('ndata( 1, 1, 1) = {} ,\n'.format(len(rn)+2))
            inlines.append("tdata(1, 1, 1, 1) = {:e} , tdata(2, 1, 1, 1) = {:e} ,\n".format(rn[0]-0.005, dn[0]))
            for i in range(len(rn)):
                inlines.append("tdata(1, {}, 1, 1) = {:e} , tdata(2, {}, 1, 1) = {:e} ,\n".format(i + 2, rn[i], i + 2, dn[i]))
            inlines.append("tdata(1, {}, 1, 1) = {:e} , tdata(2, {}, 1, 1) = {:e} ,\n".format(len(rn)+2, rn[-1] + 0.005, len(rn)+2, dn[-1]))

            if carbon:
                inlines.append('ndata( 1, 1, 3) = {} ,\n'.format(len(rn)+2))
                inlines.append("tdata(1, 1, 1, 3) = {:e} , tdata(2, 1, 1, 3) = {:e} ,\n".format(rn[0]-0.005, dn[0]))
                for i in range(len(rn)):
                    inlines.append("tdata(1, {}, 1, 3) = {:e} , tdata(2, {}, 1, 3) = {:e} ,\n".format(i + 2, rn[i], i + 2, dn[i]))
                inlines.append("tdata(1, {}, 1, 3) = {:e} , tdata(2, {}, 1, 3) = {:e} ,\n".format(len(rn)+2, rn[-1] + 0.005, len(rn)+2, dn[-1]))

                for j in range(4, 10):
                    inlines.append('ndata( 1, 1, {}) = {} ,\n'.format(j, len(rn)+2))
                    inlines.append("tdata(1, 1, 1, {}) = {:e} , tdata(2, 1, 1, {}) = {:e} ,\n".format(j, rn[0]-0.005, j, dn[0]))
                    for i in range(len(rn)):
                        inlines.append("tdata(1, {}, 1, {}) = {:e} , tdata(2, {}, 1, {}) = {:e} ,\n".format(i + 2, j, rn[i], i + 2, j,dc[i]))
                    inlines.append("tdata(1, {}, 1, {}) = {:e} , tdata(2, {}, 1, {}) = {:e} ,\n".format(len(rn) + 2, j, rn[-1] + 0.005, len(rn) + 2, j,dc[-1]))

                for j in range(3, 10):
                    inlines.append('ndata( 1, 6, {}) = {} ,\n'.format(j, len(rn)+2))
                    inlines.append("tdata(1, 1, 6, {}) = {:e} , tdata(2, 1, 6, {}) = {:e} ,\n".format(j, rn[0]-0.005, j, vrc[0]))
                    for i in range(len(rn)):
                        inlines.append("tdata(1, {}, 6, {}) = {:e} , tdata(2, {}, 6, {}) = {:e} ,\n".format(i + 2, j, rn[i], i + 2, j,vrc[i]))
                    inlines.append("tdata(1, {}, 6, {}) = {:e} , tdata(2, {}, 6, {}) = {:e} ,\n".format(len(rn)+2, j, rn[-1]+0.005, len(rn) + 2, j, vrc[-1]))

            # Heat fluxes

            inlines.append('ndata( 1, 3, 1) = {} ,\n'.format(len(rn)+2))
            inlines.append("tdata(1, 1, 3, 1) = {:e} , tdata(2, 1, 3, 1) = {:e} ,\n".format(rn[0]-0.005, ke[0]))
            for i in range(len(rn)):
                inlines.append("tdata(1, {}, 3, 1) = {:e} , tdata(2, {}, 3, 1) = {:e} ,\n".format(i + 2, rn[i], i + 2, ke[i]))
            inlines.append("tdata(1, {}, 3, 1) = {:e} , tdata(2, {}, 3, 1) = {:e} ,\n".format(len(rn)+2, rn[-1] + 0.005, len(rn)+2, ke[-1]))

            inlines.append('ndata( 1, 4, 1) = {} ,\n'.format(len(rn)+2))
            inlines.append("tdata(1, 1, 4, 1) = {:e} , tdata(2, 1, 4, 1) = {:e} ,\n".format(rn[0]-0.005, ki[0]))
            for i in range(len(rn)):
                inlines.append("tdata(1, {}, 4, 1) = {:e} , tdata(2, {}, 4, 1) = {:e} ,\n".format(i + 2, rn[i], i + 2, ki[i]))
            inlines.append("tdata(1, {}, 4, 1) = {:e} , tdata(2, {}, 4, 1) = {:e} ,\n".format(len(rn)+2, rn[-1] + 0.005, len(rn)+2, ki[-1]))
            """

        rn[0] -= delta_step
        rn[-1] += delta_step

        inlines.append('&TRANSPORT\n')
        inlines.append('ndata( 1, 1, 1) = {} ,\n'.format(len(rn)))
        for i in range(len(rn)):
            inlines.append("tdata(1, {}, 1, 1) = {:e} , tdata(2, {}, 1, 1) = {:e} ,\n".format(i+1, rn[i], i+1, dn[i]))

        if self.data['impurities']:

            for j in species_nums[1:]:
                inlines.append('ndata( 1, 1, {}) = {} ,\n'.format(j, len(rn)))
                for i in range(len(rn)):
                    inlines.append("tdata(1, {}, 1, {}) = {:e} , tdata(2, {}, 1, {}) = {:e} ,\n".format(i+1, j, rn[i], i+1, j, dc[i]))

            # C pinch term to attempt to match CER nC+6
            if np.any(vrc):
                for j in species_nums[1:]:
                    inlines.append('ndata( 1, 6, {}) = {} ,\n'.format(j, len(rn)))
                    for i in range(len(rn)):
                        inlines.append("tdata(1, {}, 6, {}) = {:e} , tdata(2, {}, 6, {}) = {:e} ,\n".format(i+1, j, rn[i], i+1, j, vrc[i]))

        # Heat fluxes

        for j in species_nums:
            inlines.append('ndata( 1, 3, {}) = {} ,\n'.format(j, len(rn)))
            for i in range(len(rn)):
                inlines.append("tdata(1, {}, 3, {}) = {:e} , tdata(2, {}, 3, {}) = {:e} ,\n".format(i+1, j, rn[i], i+1, j, ki[i]))

        inlines.append('ndata( 1, 4, 1) = {} ,\n'.format(len(rn)))
        for i in range(len(rn)):
            inlines.append("tdata(1, {}, 4, 1) = {:e} , tdata(2, {}, 4, 1) = {:e} ,\n".format(i+1, rn[i], i+1, ke[i]))

        # if carbon:
        #     # Ti is the same for all species, but n changes per species, so chi_i is set separately
        #     # Assign the same ion thermal diffusion coefficients (transport coefficient 3) from
        #     # species 1 to all other ion species
        #     # This seems to be incorrect, it breaks the reading
        #     for i in range(3, 9):
        #         inlines.append('addspec( {}, 3, 1) = {} ,\n'.format(i, i))

        inlines.append('no_pflux = .true.\n')  # Will use whatever is in b2.transport.parameters for PFR
        inlines.append('/\n')
        
        # Write out file
        
        with open(inFile, 'w') as f:
            for i in range(len(inlines)):
                f.write(inlines[i])

