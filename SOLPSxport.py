"""
This class is used by routines in 'SOLPS_match_upstream.py' to read experimental data,
read SOLPS data, and then calculate the updated radial transport coefficients to attempt
to match SOLPS to the experimental profiles

R.S. Wilcox, J.M. Canik and J.D. Lore 2020
contact: wilcoxr@fusion.gat.com
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import SOLPSutils as sut

plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.facecolor':'w'})
plt.rcParams.update({'mathtext.default': 'regular'})

eV = 1.60217662e-19


class SOLPSxport:

    def __init__(self, workdir, gfile_loc, carbon_bool = True, b2plot_dev_x11 = False):
        """
        Make sure you source setup.csh or setup.ksh before running any of this!
        
        Inputs:
          workdir         Directory with the SOLPS outputs
          gfile_loc       location of corresponding g file
          carbon_bool     Set to False if carbon is not in the run
          b2plot_dev_x11  Set to True if you want a figure to pop up for every b2plot call
        """
        shot_loc_in_gfile_string = gfile_loc.rfind('g')
        shot = int(gfile_loc[shot_loc_in_gfile_string+1 : shot_loc_in_gfile_string+7])

        shot_ind_in_workdir = workdir.rfind(str(shot))
        if shot_ind_in_workdir > 0:
            workdir_short = workdir[shot_ind_in_workdir:]
        else:
            workdir_short = None
            
        self.data = {'workdir':workdir, 'workdir_short':workdir_short, 'gfile_loc': gfile_loc,
                     'pedData':{}, 'solpsData':{'profiles':{}}, 'carbon':carbon_bool, 'shot':shot}

        if (not b2plot_dev_x11) and os.environ['B2PLOT_DEV'] == 'x11 ps':
            print("Changing environment variable B2PLOT_DEV to 'ps'")
            os.environ['B2PLOT_DEV'] = 'ps'
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
            print("Calling '2d_profiles' in directory: " + working)
            os.system('2d_profiles')

        rx, ne_ = sut.readProf('ne3da.last10')
        rx, dn_ = sut.readProf('dn3da.last10')
        rx, te_ = sut.readProf('te3da.last10')
        rx, ke_ = sut.readProf('ke3da.last10')
        rx, ti_ = sut.readProf('ti3da.last10')
        rx, ki_ = sut.readProf('ki3da.last10')
        
        os.chdir(olddir)

        # Cast everything as np array so it doesn't break later when performing math operations
        
        ne = np.array(ne_)
        dn = np.array(dn_)
        te = np.array(te_)
        ke = np.array(ke_)
        ti = np.array(ti_)
        ki = np.array(ki_)

        last10_dic = {'rx':rx,'ne':ne,'dn':dn,'te':te,'ke':ke,'ti':ti,'ki':ki}
    
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
        Either (1) provide the location of the saved profiles file (prefer using json module, previously used pickle)
                   (should have extension *.pkl for now, json not set up yet)
            or (2) give info to retrieve it from MDSplus

            (1) can be written using getProfDBPedFit from SOLPSutils.py (easily reproduced manually if needed)
            (2) requires access to atlas.gat.com
        """
        
        if profiles_file is None:
            if verbose: print('Getting profile fit data from MDSplus server on atlas.gat.com')
            
            self.timeid = timeid
            self.data['pedData']['fitVals'] = sut.getProfDBPedFit(shotnum, timeid, runid)
            
        else:
            if verbose: print('Reading profile fit data from saved file: ' + profiles_file)

            if profiles_file[-4:] == '.pkl':
                import pickle

                with open(profiles_file, 'rb') as f:
                    self.data['pedData']['fitVals'] = pickle.load(f)

            else:
                import json

                self.data['pedData']['fitVals'] = json.load(open(profiles_file))

    # ----------------------------------------------------------------------------------------
    
    def load_pfile(self, pfile_loc, plotit = False):
        self.timeid = pfile_loc[pfile_loc.rfind('.')+1:]

        pfile_dict = sut.read_pfile(pfile_loc)

        psin = pfile_dict['psinorm']
        ne = pfile_dict['ne(10^20/m^3)'] * 10  # needs to be in units of 10^19/m^3
        Te = pfile_dict['te(KeV)'] * 1e3
        try:
            Ti = pfile_dict['ti(KeV)'] * 1e3
        except:
            Ti = Te
            
        self.data['pedData']['pfile'] = pfile_dict
        
        if plotit:
            SOLPS_profs = self.data['solpsData']['last10']
            rne_SOLPS = SOLPS_profs['rne']
            rte_SOLPS = SOLPS_profs['rte']
            rti_SOLPS = SOLPS_profs['rti']
            ne_SOLPS = SOLPS_profs['ne']
            te_SOLPS = SOLPS_profs['te']
            ti_SOLPS = SOLPS_profs['ti']
            
            
            f, ax = plt.subplots(3, sharex = 'all')
            ax[0].plot(np.array(rne_SOLPS) + 1, ne_SOLPS * 1e-19, '-kx', lw=2, label='ne_SOLPS')
            ax[0].plot(psin, ne, '--r', lw = 2, label = 'ne')
            ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
            ax[0].legend(loc = 'best')
            ax[0].grid('on')

            ax[1].plot(np.array(rte_SOLPS) + 1, te_SOLPS, '-kx', lw = 2, label = 'Te_SOLPS')
            ax[1].plot(psin, Te, '--r', lw = 2, label = 'Te')
            ax[1].set_ylabel('Te (eV)')
            ax[1].legend(loc = 'best')
            ax[1].grid('on')
            
            ax[2].plot(np.array(rti_SOLPS) + 1, ti_SOLPS, '-kx', lw = 2, label = 'Ti_SOLPS')
            ax[2].plot(psin, Ti, '--r', lw = 2, label = 'Ti')
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

        Inputs:
          XXmod   Fit used for each parameter
          npsi    Number of points to populate full radial fit, from psin=0 (linearly distributed)
          psinMax End of radial grid to populate, in units of psin
                  (this defaults to be slightly larger than the SOLPS grid if nothing is given)
        """
        if psinMax is None:
            if 'psiSOLPS' in self.data['solpsData'].keys():
                psinMax = np.max(self.data['solpsData']['psiSOLPS']) + 0.001
            else:
                psinMax = 1.02

        psiProf = np.linspace(0, psinMax, npsi+1)
        
        self.data['pedData']['fitPsiProf'] = psiProf
        self.data['pedData']['fitProfs'] = {}
    
        # !!! Need to subtract shift from psiProf to move separatrix for power balance !!!

        # make ne profile
        if nemod == 'tnh0':
            necoef = self.data['pedData']['fitVals']['netnh0psi']['y']
            neprof = sut.calcTanhMulti(necoef,psiProf)
        elif nemod == 'tanh':
            necoef = self.data['pedData']['fitVals']['netanhpsi']['y']
            neprof = sut.calcTanhMulti(necoef,psiProf)
        self.data['pedData']['fitProfs']['neprof'] = neprof
            
        # make Te profile
        if temod == 'tnh0':
            tecoef = self.data['pedData']['fitVals']['tetnh0psi']['y']
            teprof = sut.calcTanhMulti(tecoef,psiProf)
        elif nemod == 'tanh':
            tecoef = self.data['pedData']['fitVals']['tetanhpsi']['y']
            teprof = sut.calcTanhMulti(tecoef,psiProf)
        self.data['pedData']['fitProfs']['teprof'] = teprof
        
        if ncmod == 'spl':
            zfzpsi = self.data['pedData']['fitVals']['zfz1splpsi']['x']
            zfzprof = self.data['pedData']['fitVals']['zfz1splpsi']['y']
            zfzfunc = interp1d(zfzpsi, zfzprof, bounds_error = False,
                               fill_value = zfzprof[np.argmax(zfzpsi)])
            # extrapolate to psin>1 using highest value of psin available
        
            self.data['pedData']['fitProfs']['ncprof'] = zfzfunc(psiProf) * neprof / 6
            
        elif ncmod == 'tanh':
            nccoef = self.data['pedData']['fitVals']['nztanhpsi']['y']
            ncprof = sut.calcTanhMulti(nccoef,psiProf)
            self.data['pedData']['fitProfs']['ncprof'] = ncprof * neprof / 6
            
        if plotit:
            f, ax = plt.subplots(2, sharex = 'all')

            ax[0].plot(psiProf, neprof, '-b', lw = 2, label = 'n$_e$')
            ax[0].plot(psiProf, 6*self.data['pedData']['fitProfs']['ncprof'], '--k', lw = 2, label = '6*n$_C$')
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

    def modify_ti(self, ratio_fileloc = '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt',
                  sol_points = None, max_psin = 1.1, decay_length = 0.015,
                  rad_loc_for_exp_decay = 1.0, reduce_ti = True, ti_min = 1, plotit = False):
        """
        Manually modify the Ti profile to be more realistic

        CER measures C+6, the temperature of which can differ significantly from
        the main ion species in the edge.

        Inputs:
          sol_points   Number of extra points to add in the SOL
          max_psi      sol_points will be evenly distributed between rad_loc_for_exp_decay
                       and max_psi
          decay_length Decay length for exponential falloff imposed into SOL (in units of psin)
          reduce_ti    Use a profile from a single comparison case of T_D vs T_C+6
                       to reduce the "measured" value of T_D to be more realistic
          ti_min       Ti decays exponentially to this value (in eV)
        """

        tiexp = self.data['pedData']['fitVals']['tisplpsi']['y']
        tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']

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
                                        np.array(T_DC_ratio), left=1)
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
            psi_TS = self.data['pedData']['fitPsiProf']
            teexp = self.data['pedData']['fitProfs']['teprof']

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

        self.data['pedData']['fitVals']['ti_mod'] = {'x':xrad, 'y':ti_mod}

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
        
        if 'fitVals' not in self.data['pedData']:
            self.data['pedData']['fitVals'] = {'nedatpsi':{}, 'zfz1datpsi':{}, 'tisplpsi':{}}
        
        # Set 'fitProfs' values (linearly interpolate if psi locations from prof files are no good)
        
        if (all(psin[f] == psin[fit]) for f in profs) and len(psin[fit]) >= min_npsi:
            
            # Deuterium profiles
            
            self.data['pedData']['fitPsiProf'] = psin[fit]
            self.data['pedData']['fitProfs'] = {'neprof': vals['n_e'] / 1.0e20,
                                                'teprof': vals['T_e'] / 1.0e3}
            # keep units consistent with Tom's tools

            self.data['pedData']['fitVals']['tisplpsi']['x'] = psin[fit]
            if 'T_D' in vals.keys():
                self.data['pedData']['fitVals']['tisplpsi']['y'] = vals['T_D'] / 1.0e3
            else:   # assume T_D = T_C
                self.data['pedData']['fitVals']['tisplpsi']['y'] = vals['T_12C6'] / 1.0e3
            
            # Carbon profiles
            
            if self.data['carbon']:
                self.data['pedData']['fitVals']['nedatpsi']['x'] = psin[fit]
                self.data['pedData']['fitVals']['nedatpsi']['y'] = vals['n_e'] / 1.0e20
                
                self.data['pedData']['fitVals']['zfz1datpsi']['x'] = psin[fit]
                self.data['pedData']['fitVals']['zfz1datpsi']['y'] = 6*vals['n_12C6']/vals['n_e']

                self.data['pedData']['fitProfs']['ncprof'] = vals['n_12C6']
            
        else:
            
            psiProf = np.linspace(0, psiMax, min_npsi + 1)
    
            self.data['pedData']['fitPsiProf'] = psiProf
            vals_interp = {}
            for fit in profs:
                fitfunc = interp1d(psin[fit], vals[fit])
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

    def calcPsiVals(self, plotit = False):
        """
        Call b2plot to get the locations of each grid cell in psin space

        Saves the values to dictionaries in self.data['solpsData']
        """
        from scipy import interpolate

        """
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

        dsa, crLowerLeft = sut.B2pl('0 crx writ jxa f.y', wdir = wdir)
        # dummy, crLowerRight = B2pl('1 crx writ jxa f.y', wdir = wdir)
        # Only 2 unique psi values per cell, grab 0 and 2
        dummy, crUpperLeft = sut.B2pl('2 crx writ jxa f.y', wdir = wdir)  # all x inds are the same
        dummy, czLowerLeft = sut.B2pl('0 cry writ jxa f.y', wdir = wdir)
        dummy, czUpperLeft = sut.B2pl('2 cry writ jxa f.y', wdir = wdir)
        ncells = len(dummy)

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
            psiN_copy = psiN.copy()
            psiN_copy[psiN > 1.01] = np.nan
            psin_masked = np.ma.masked_invalid(psiN_copy)

            plt.figure()
            plt.pcolormesh(gR, gZ, psin_masked, cmap = 'inferno')
            plt.colorbar()
            plt.plot(g['rlim'], g['zlim'], 'k', lw = 2)
            plt.contour(gR, gZ, psiN, [1], colors = 'k')
            plt.title('$\psi_N$ from g file')
            plt.axis('equal')
            plt.xlim([np.min(gR), np.max(gR)])
            plt.ylim([np.min(gZ), np.max(gZ)])
            plt.plot(R_solps_top, Z_solps_top, 'g', lw = 3)
            plt.plot([1.94, 1.94], [-1.5, 1.5], '--k', lw=1)  # Thomson laser path

            plt.figure()
            plt.plot(R_solps_top, psi_solps, 'k', lw = 2)
            plt.xlabel('R at midplane (m)')
            plt.ylabel('$\psi_N$')

            plt.show(block = False)
    
    # ----------------------------------------------------------------------------------------
    
    def getSOLPSfluxProfs(self, plotit = False):
        """
        Calls b2plot to get the particle flux profiles
        """
        # x variable is identical for all of these
        x_fTot, fluxTot = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ jxa f.y")
        x_fTot, fluxD = sut.B2pl("fnay 1 zsel sy m/ writ jxa f.y")
        dummy, fluxConv = sut.B2pl("na za m* vlay m* 0 0 sumz writ jxa f.y")
        dummy, na = sut.B2pl("na 0 0 sumz writ jxa f.y")
        # dummy, hy1 = sut.B2pl("hy1 writ jxa f.y")  # not used anymore
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
        # self.data['solpsData']['profiles']['hy1'] = np.array(hy1)  # not used anymore
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
    
    def getSOLPSCarbonProfs(self, plotit = False, verbose=True):
        """
        Calls b2plot to get the carbon profiles
        """

        x_nc, nc_solps = sut.B2pl("na 8 zsel psy writ jxa f.y")
        x_nd, nd_solps = sut.B2pl("na 1 zsel psy writ jxa f.y")
        dummy, flux_carbon = sut.B2pl("fnay 8 zsel psy writ jxa f.y")  # x variables are the same
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
            if 'ncprof' in self.data['pedData']['fitProfs'].keys():
                nc_psi = self.data['pedData']['fitPsiProf']
                nc_prof = np.array(self.data['pedData']['fitProfs']['ncprof']*100)  # in  m^-3
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
                      Dn_max = 10, chie_max = 200, chii_max = 200, vrc_mag=0.0, ti_decay_len = 0.015,
                      reduce_Ti_fileloc = '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt',
                      use_ratio_bc = True, debug_plots = False, verbose = False, figblock = False):
        """
        Calculates the transport coefficients to be written into b2.transport.inputfile
        
        Requires experimental profiles to have already been saved to self.data

        Inputs:
          ti_decay_len: Decay length for exponential falloff outside separatrix (units of psin)
                        (set to None to skip this)
          reduce_Ti_fileloc: Location of a saved array to get the ratio between T_C (measured) and T_i
                        This ratio was calculated from Shaun Haskey's T_D measurements
                        for 171558 @ 3200 ms
                        Set to None to use Ti from CER C+6
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
        if self.data['carbon']:
            ndold = self.data['solpsData']['profiles']['nD']
        else:
            ndold = neold
        
        fluxTot = self.data['solpsData']['profiles']['fluxTot']
        fluxD = self.data['solpsData']['profiles']['fluxD']
        fluxConv = self.data['solpsData']['profiles']['fluxConv']
        # hy1 = self.data['solpsData']['profiles']['hy1']  # Not used here
        qe = self.data['solpsData']['profiles']['qe']
        qi = self.data['solpsData']['profiles']['qi']
        
        psi_to_dsa_func = interp1d(psi_solps, dsa, fill_value = 'extrapolate')

        # Convective portion of heat flux to be subtracted to get diffusive component
        # These are not actually used with the way it's coded now
        # SOLPS_qe_conv = 2.5 * dold * teold * eV
        # SOLPS_qi_conv = 2.5 * dold * tiold * eV
        
        
        # ne and Gamma_e

        psi_data_fit = self.data['pedData']['fitPsiProf']
        neexp = 1.0e20 * self.data['pedData']['fitProfs']['neprof']
        
        dsa_TSprofile = psi_to_dsa_func(psi_data_fit)

        gnold_dsa = np.gradient(neold) / np.gradient(dsa)  # Only used for dnew_ratio
        gnexp_dsa = np.gradient(neexp) / np.gradient(dsa_TSprofile)

        gnexp_dsafunc = interp1d(dsa_TSprofile, gnexp_dsa)
        # psi_to_dsa_func function only valid in SOLPS range,
        # so gnexp_dsafunc shouldn't be applied outside that
        gnexp_solpslocs_dsa = gnexp_dsafunc(dsa)

        # Set boundary condition to get ne[-1] right
        expden_dsa_func = interp1d(dsa_TSprofile, neexp)
        den_decay_len = (expden_dsa_func(dsa[-2]) - expden_dsa_func(dsa[-1])) / \
            np.mean([expden_dsa_func(dsa[-1]), expden_dsa_func(dsa[-2])])
        if verbose: print('den_decay_len = ' + str(den_decay_len))
        gnexp_solpslocs_dsa[-1] = -expden_dsa_func(dsa[-1]) / den_decay_len

        # this method assumes no convective transport (ok in some cases)
        dnew_ratio = (gnold_dsa / gnexp_solpslocs_dsa) * dold


        flux = fluxD - fluxConv  # Conductive portion of the total flux
        dnew_flux = -flux / gnexp_solpslocs_dsa

        if use_ratio_bc:
            dnew_ratio[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
            dnew_flux[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
        

        dnew_ratio[0] = dnew_ratio[1]
        dnew_flux[0] = dnew_flux[1]
        

        # Te and ke
        
        teexp = 1.0e3 * self.data['pedData']['fitProfs']['teprof']
        
        gteold = np.gradient(teold) / np.gradient(dsa)
        gteexp = np.gradient(teexp) / np.gradient(dsa_TSprofile)

        gteexp_dsafunc = interp1d(dsa_TSprofile, gteexp)
        gteexp_solpslocs = gteexp_dsafunc(dsa)
        
        # Set boundary condition to get Te[-1] right
        expTe_dsa_func = interp1d(dsa_TSprofile, teexp)
        te_decay_len = (expTe_dsa_func(dsa[-2]) - expTe_dsa_func(dsa[-1])) / \
            np.mean([expTe_dsa_func(dsa[-1]), expTe_dsa_func(dsa[-2])])
        if verbose: print('Te_decay_len = ' + str(te_decay_len))
        gteexp_solpslocs[-1] = -expTe_dsa_func(dsa[-1]) / te_decay_len
        
        
        kenew_ratio = (gteold / gteexp_solpslocs) * keold

        # gradient has to be in dsa to work
        kenew_flux = -(qe - 2.5 * fluxTot * teold * eV) / (neold * eV * gteexp_solpslocs)

        if use_ratio_bc:
            kenew_ratio[-1] = keold[-1] * teold[-1] / expTe_dsa_func(dsa[-1])
            kenew_flux[-1] = keold[-1] * teold[-1] / expTe_dsa_func(dsa[-1])

        kenew_ratio[0] = kenew_ratio[1]   # gaurd cells
        kenew_flux[0] = kenew_flux[1]
        
        # Ti and ki

        if reduce_Ti_fileloc or (ti_decay_len is not None):
            self.modify_ti(ratio_fileloc = reduce_Ti_fileloc, sol_points = 10, max_psin = np.max(psi_solps) + 0.001,
                           decay_length = ti_decay_len, rad_loc_for_exp_decay = 1.0,
                           plotit = debug_plots, reduce_ti = (reduce_Ti_fileloc is not None))

            tiexp = 1.0e3*self.data['pedData']['fitVals']['ti_mod']['y']
            tiexppsi = self.data['pedData']['fitVals']['ti_mod']['x']

        else:
            tiexp = 1.0e3*self.data['pedData']['fitVals']['tisplpsi']['y']
            tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']
        
        dsa_tiprofile = psi_to_dsa_func(tiexppsi)
        
        gtiold = np.gradient(tiold) / np.gradient(dsa)
        gtiexp = np.gradient(tiexp) / np.gradient(dsa_tiprofile)
        

        gtiexp_dsafunc = interp1d(dsa_tiprofile, gtiexp, kind='linear', fill_value = 'extrapolate')

        gtiexp_solpslocs = gtiexp_dsafunc(dsa)
        
        # Set boundary condition to get Ti[-1] right
        expTi_dsa_func = interp1d(dsa_tiprofile, tiexp, fill_value = 'extrapolate')
        if ti_decay_len is not None:
            gtiexp_solpslocs[-1] = -expTi_dsa_func(dsa[-1]) / ti_decay_len
        
        
        kinew_ratio = (gtiold / gtiexp_solpslocs) * kiold

        # gradient has to be in dsa to work
        kinew_flux = -(qi - 2.5 * fluxTot * tiold * eV) / (ndold * eV * gtiexp_solpslocs)

        if use_ratio_bc:
            kinew_ratio[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])
            kinew_flux[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])

        kinew_ratio[0] = kinew_ratio[1]   # gaurd cells
        kinew_flux[0] = kinew_flux[1]
        
        
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
        
        if self.data['carbon']:
            # vrc_mag = 20.0  # 60 for gauss
            vr_pos = 0.97
            # vr_wid = 0.02
            # sig = vr_wid / 2.3548
            # vr_shape = np.exp(-((psi_solps - vr_pos) ** 2) / (2 * sig ** 2))  # Gaussian
            
            vr_shape = psi_solps - vr_pos  # linear
            
            vr_carbon = vr_shape * vrc_mag / max(vr_shape)
        
            D_carbon = Dn_min + 0.0 * dnew_flux
            # D_carbon[19:] = dnew_flux[19:]
            D_carbon[1:] = dnew_flux[1:]
                    
        else:
            vr_carbon = None
            D_carbon = None

        coef_limits = {'Dn_min':Dn_min, 'Dn_max':Dn_max, 'chie_min':chie_min,
                       'chii_min':chii_min, 'chie_max':chie_max, 'chii_max':chii_max}


        self.data['solpsData']['xportCoef'] = {'dnew_ratio': dnew_ratio, 'dnew_flux': dnew_flux,
                                               'kenew_ratio': kenew_ratio, 'kenew_flux':kenew_flux,
                                               'kinew_ratio': kinew_ratio, 'kinew_flux':kinew_flux,
                                               'vr_carbon': vr_carbon, 'D_carbon': D_carbon,
                                               'limits': coef_limits}
        if plotit:
            self.plotXportCoef(figblock = figblock)

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
            plt.plot(tiexppsi, 1.0e3 * self.data['pedData']['fitVals']['tisplpsi']['y'],
                     'k', lw=2, label='Ti measured')
            plt.plot(tiexppsi, tiexp, 'r', label='Ti_new')
            plt.xlabel('$\psi_N$')
            plt.ylabel('T$_i$ (keV)')
            plt.legend(loc='best')

            if self.data['carbon']:
                plt.figure()
                plt.plot(psi_solps, dnew_ratio, '-or', label='dnew_ratio')
                plt.plot(psi_solps, dnew_flux, '-ob', label='dnew_flux')
                plt.plot(psi_solps, D_carbon, '-xg', label='D_carbon')
                plt.legend(loc='best')

            plt.show(block=figblock)

    # ----------------------------------------------------------------------------------------

    def plotXportCoef(self, figblock=False):
        """
        Plot the upstream profiles from SOLPS compared to the experiment
        along with the corresponding updated transport coefficients
        """

        dnew_ratio = self.data['solpsData']['xportCoef']['dnew_ratio']
        dnew_flux = self.data['solpsData']['xportCoef']['dnew_flux']
        kenew_ratio = self.data['solpsData']['xportCoef']['kenew_ratio']
        kenew_flux = self.data['solpsData']['xportCoef']['kenew_flux']
        kinew_ratio = self.data['solpsData']['xportCoef']['kinew_ratio']
        kinew_flux = self.data['solpsData']['xportCoef']['kinew_flux']
        coef_limits = self.data['solpsData']['xportCoef']['limits']

        psi_data_fit = self.data['pedData']['fitPsiProf']
        neexp = 1.0e20 * self.data['pedData']['fitProfs']['neprof']
        teexp = 1.0e3 * self.data['pedData']['fitProfs']['teprof']
        if 'ti_mod' in self.data['pedData']['fitVals'].keys():
            tiexp = 1.0e3*self.data['pedData']['fitVals']['ti_mod']['y']
            tiexppsi = self.data['pedData']['fitVals']['ti_mod']['x']
        else:
            tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']
            tiexp = 1.0e3*self.data['pedData']['fitVals']['tisplpsi']['y']


        psi_solps = self.data['solpsData']['psiSOLPS']
        neold = self.data['solpsData']['last10']['ne']
        dold = self.data['solpsData']['last10']['dn']
        teold = self.data['solpsData']['last10']['te']
        keold = self.data['solpsData']['last10']['ke']
        tiold = self.data['solpsData']['last10']['ti']
        kiold = self.data['solpsData']['last10']['ki']


        # Find limits for plots
        TS_inds_in_range = np.where(psi_data_fit > np.min(psi_solps))[0]
        Ti_inds_in_range = np.where(tiexppsi > np.min(psi_solps))[0]
        max_Te = np.max([np.max(teold), np.max(teexp[TS_inds_in_range])]) / 1.0e3
        max_Ti = np.max([np.max(tiold), np.max(tiexp[Ti_inds_in_range])]) / 1.0e3
        max_ne = np.max([np.max(neold), np.max(neexp[TS_inds_in_range])]) / 1.0e19
        max_dn = np.max([np.max(dold), np.max(dnew_ratio), np.max(dnew_flux)])
        max_ke = np.max([np.max(keold), np.max(kenew_ratio), np.max(kenew_flux)])
        max_ki = np.max([np.max(kiold), np.max(kinew_ratio), np.max(kinew_flux)])
        min_dn = np.min([np.min(dold), np.min(dnew_ratio), np.min(dnew_flux)])
        min_ke = np.min([np.min(keold), np.min(kenew_ratio), np.min(kenew_flux)])
        min_ki = np.min([np.min(kiold), np.min(kinew_ratio), np.min(kinew_flux)])


        headroom = 1.05
        xlims = [np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01]

        f, ax = plt.subplots(2, 3, sharex = 'all')
        ax[0, 0].plot(psi_data_fit, neexp / 1.0e19, '--bo', lw = 1, label = 'TS data')
        ax[0, 0].plot(psi_solps, neold / 1.0e19, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0, 0].legend(loc = 'best', fontsize=14)
        ax[0, 0].set_ylim([0, max_ne*headroom])
        ax[0, 0].grid('on')

        ax[1, 0].semilogy(psi_solps, dnew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 0].semilogy(psi_solps, dnew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 0].semilogy(psi_solps, dold, '-xr', lw = 2, label = 'old')
        if coef_limits['Dn_min'] is not None:
            ax[1, 0].semilogy(xlims, [coef_limits['Dn_min'], coef_limits['Dn_min']], '--m')
        if coef_limits['Dn_max'] is not None:
            ax[1, 0].semilogy(xlims, [coef_limits['Dn_max'], coef_limits['Dn_max']], '--m')
        ax[1, 0].set_ylabel('D (m$^2$/s)')
        ax[1, 0].set_xlabel('$\psi_N$')
        ax[1, 0].set_ylim([min_dn/np.sqrt(headroom), max_dn*headroom])
        ax[1, 0].grid('on')

        ax[0, 1].plot(psi_data_fit, teexp / 1.0e3, '--bo', lw = 1, label = 'Data')
        ax[0, 1].plot(psi_solps, teold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 1].set_ylabel('T$_e$ (keV)')
        ax[0, 1].set_ylim([0, max_Te*headroom])
        ax[0, 1].grid('on')
        if self.data['workdir_short'] is not None:
            ax[0, 1].set_title(self.data['workdir_short'], fontsize=10)

        ax[1, 1].semilogy(psi_solps, kenew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 1].semilogy(psi_solps, kenew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 1].semilogy(psi_solps, keold, '-xr', lw = 2, label = 'old')
        if coef_limits['chie_min'] is not None:
            ax[1, 1].semilogy(xlims, [coef_limits['chie_min'], coef_limits['chie_min']], '--m')
        if coef_limits['chie_max'] is not None:
            ax[1, 1].semilogy(xlims, [coef_limits['chie_max'], coef_limits['chie_max']], '--m')
        ax[1, 1].set_ylabel('$\chi_e$ (m$^2$/s)')
        ax[1, 1].set_xlabel('$\psi_N$')
        ax[1, 1].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        ax[1, 1].set_ylim([min_ke/np.sqrt(headroom), max_ke*headroom])
        ax[1, 1].grid('on')

        ax[0, 2].plot(psi_solps, tiold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 2].plot(tiexppsi, tiexp / 1.0e3, '--bo', lw = 1, label = 'Data')
        ax[0, 2].set_ylabel('T$_i$ (keV)')
        ax[0, 2].set_ylim([0, max_Ti*headroom])
        ax[0, 2].grid('on')

        ax[1, 2].semilogy(psi_solps, kinew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 2].semilogy(psi_solps, kinew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 2].semilogy(psi_solps, kiold, '-xr', lw = 2, label = 'old')
        if coef_limits['chii_min'] is not None:
            ax[1, 2].semilogy(xlims, [coef_limits['chii_min'], coef_limits['chii_min']], '--m')
        if coef_limits['chii_max'] is not None:
            ax[1, 2].semilogy(xlims, [coef_limits['chii_max'], coef_limits['chii_max']], '--m')
        ax[1, 2].set_ylabel('$\chi_i$ (m$^2$/s)')
        ax[1, 2].set_xlabel('$\psi_N$')
        ax[1, 2].set_xlim(xlims)
        ax[1, 2].set_ylim([min_ki/np.sqrt(headroom), max_ki*headroom])
        ax[1, 2].grid('on')
        ax[1, 2].legend(loc='best', fontsize=12)

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

        psi_data_fit = self.data['pedData']['fitPsiProf']
        nefit = 1.0e20 * self.data['pedData']['fitProfs']['neprof']
        tefit = self.data['pedData']['fitProfs']['teprof']
        tifit = self.data['pedData']['fitVals']['tisplpsi']['y']
        tifitpsi = self.data['pedData']['fitVals']['tisplpsi']['x']

        rawdat_keys = ['nedatpsi', 'tedatpsi']
        rawdat_scalars = [10, 1.0]  # ne saved as 10^20, we want 10^19
        if include_ti:
            rawdat_keys.append('tidatpsi')
            rawdat_scalars.append(1.0)
        nprofs = len(rawdat_keys)

        # Find limits of Te, Ti for plots
        TS_inds_in_range = np.where(psi_data_fit > np.min(psi_solps))[0]
        Ti_inds_in_range = np.where(tifitpsi > np.min(psi_solps))[0]
        max_ne = np.max([np.max(nesolps), np.max(nefit[TS_inds_in_range])]) / 1.0e19
        max_Te = np.max([np.max(tesolps), np.max(tefit[TS_inds_in_range])])
        max_Ti = np.max([np.max(tisolps), np.max(tifit[Ti_inds_in_range])])

        f, ax = plt.subplots(nprofs, sharex = 'all')

        for i in range(nprofs):
            ax[i].errorbar(self.data['pedData']['fitVals'][rawdat_keys[i]]['x'],
                           self.data['pedData']['fitVals'][rawdat_keys[i]]['y'] * rawdat_scalars[i],
                           self.data['pedData']['fitVals'][rawdat_keys[i]]['yerr'] * rawdat_scalars[i],
                           xerr=None, fmt='o', ls='', c='k', mfc='None', mec='k',
                           zorder=1, label='Experimental Data')

        ax[0].plot(psi_data_fit, nefit / 1.0e19, '--k', lw=2, zorder=3, label='Experimental Fit')
        # ax[0].plot(psi_solps, nesolps / 1.0e19, 'xr', lw=2, mew=2, ms=10, label='SOLPS')
        ax[0].plot(psi_solps, nesolps / 1.0e19, '-r', lw=2, zorder=2, label='SOLPS')
        ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0].legend(loc='best', fontsize=14)
        ax[0].set_ylim([0, max_ne * headroom])

        ax[1].plot(psi_data_fit, tefit, '--k', lw=2, zorder=3, label='Experimental Fit')
        # ax[1].plot(psi_solps, tesolps, 'xr', mew=2, ms=10, label='SOLPS')
        ax[1].plot(psi_solps, tesolps, '-r', lw=2, zorder=2, label='SOLPS')
        ax[1].set_ylabel('T$_e$ (keV)')
        ax[1].set_ylim([0, max_Te * headroom])
        ax[1].set_yticks(np.arange(0, max_Te * headroom + 0.2, 0.2))

        if include_ti:
            # ax[2].plot(psi_solps, tisolps, 'xr', mew = 2, ms = 10, label = 'SOLPS')
            ax[2].plot(tifitpsi, tifit, '--k', lw = 2, zorder=3, label = 'Experimental Fit')
            ax[2].plot(psi_solps, tisolps, '-r', lw = 2, zorder=2, label = 'SOLPS')

            if 'ti_mod' in self.data['pedData']['fitVals'].keys():
                ax[2].plot(self.data['pedData']['fitVals']['ti_mod']['x'],
                           self.data['pedData']['fitVals']['ti_mod']['y'],
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

        psi_data_fit = self.data['pedData']['fitPsiProf']
        nefit = 1.0e20 * self.data['pedData']['fitProfs']['neprof']
        tefit = self.data['pedData']['fitProfs']['teprof']
        tifit = self.data['pedData']['fitVals']['tisplpsi']['y']
        tifitpsi = self.data['pedData']['fitVals']['tisplpsi']['x']

        rawdat_keys = ['nedatpsi', 'tedatpsi']
        rawdat_scalars = [10, 1.0]  # ne saved as 10^20, we want 10^19
        if include_ti:
            rawdat_keys.append('tidatpsi')
            rawdat_scalars.append(1.0)
        nprofs = len(rawdat_keys)

        # Find limits of Te, Ti for plots
        TS_inds_in_range = np.where((self.data['pedData']['fitVals']['nedatpsi']['x'] > np.min(psi_solps)) &
                                    (self.data['pedData']['fitVals']['nedatpsi']['x'] < np.max(psi_solps)))[0]
        Ti_inds_in_range = np.where((self.data['pedData']['fitVals']['tidatpsi']['x'] > np.min(psi_solps)) &
                                    (self.data['pedData']['fitVals']['tidatpsi']['x'] < np.max(psi_solps)))[0]
        max_raw_ne = np.max(self.data['pedData']['fitVals']['nedatpsi']['y'][TS_inds_in_range] +
                            self.data['pedData']['fitVals']['nedatpsi']['yerr'][TS_inds_in_range])
        max_raw_te = np.max(self.data['pedData']['fitVals']['tedatpsi']['y'][TS_inds_in_range] +
                            self.data['pedData']['fitVals']['tedatpsi']['yerr'][TS_inds_in_range])
        max_raw_ti = np.max(self.data['pedData']['fitVals']['tidatpsi']['y'][Ti_inds_in_range] +
                            self.data['pedData']['fitVals']['tidatpsi']['yerr'][Ti_inds_in_range])
        max_ne = np.max([np.max(nesolps) / 1.0e19, max_raw_ne *10])
        max_Te = np.max([np.max(tesolps), max_raw_te])
        max_Ti = np.max([np.max(tisolps), max_raw_ti])
        max_temp = np.max([max_Te, max_Ti])  # just use this so they're on the same scale

        f, ax = plt.subplots(2, nprofs, sharex='all')

        for i in range(nprofs):
            ax[0, i].errorbar(self.data['pedData']['fitVals'][rawdat_keys[i]]['x'],
                              self.data['pedData']['fitVals'][rawdat_keys[i]]['y'] * rawdat_scalars[i],
                              self.data['pedData']['fitVals'][rawdat_keys[i]]['yerr'] * rawdat_scalars[i],
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

            if 'ti_mod' in self.data['pedData']['fitVals'].keys():
                ax[0, 2].plot(self.data['pedData']['fitVals']['ti_mod']['x'],
                              self.data['pedData']['fitVals']['ti_mod']['y'],
                              '--b', mew=2, zorder=4, label='Modified Ti fit')
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
    
    def writeXport(self, new_filename = 'b2.transport.inputfile_new', solps5_0 = False,
                   scale_D = 1, ke_use_grad = False, ki_use_grad = False):
        """
        Write the b2.transport.inputfile using values saved in this object

        Inputs:
          ke/i_use_grad      Use ratio of the gradients for new values of chi_i/e
          scale_D            Scalar factor to modify all particle diffusion coefficients
                             (when going from density BC to flux BC, need to reduce the transport by a
                             factor proportional to the difference in core flux between the two cases)
        """
        
        wdir = self.data['workdir']
        inFile = os.path.join(wdir, new_filename)
        if os.path.isfile(inFile):
            print("'" + new_filename + "' already exists, renaming existing " +
                  "file to 'old_xport_coef' and writing new one")
            movedFile = os.path.join(wdir, 'old_xport_coef')
            cmds = 'cp ' + inFile + ' ' + movedFile
            os.system(cmds)
        
        rn = self.data['solpsData']['last10']['rx']
        dn = self.data['solpsData']['xportCoef']['dnew_flux'] * scale_D
        if ke_use_grad:
            ke = self.data['solpsData']['xportCoef']['kenew_ratio']
        else:
            ke = self.data['solpsData']['xportCoef']['kenew_flux']
        if ki_use_grad:
            ki = self.data['solpsData']['xportCoef']['kinew_ratio']
        else:
            ki = self.data['solpsData']['xportCoef']['kinew_flux']
        vrc = self.data['solpsData']['xportCoef']['vr_carbon']
        dc = self.data['solpsData']['xportCoef']['D_carbon'] * scale_D
        carbon = self.data['carbon']
        
        # Step the boundary points out a tiny bit so that they are
        # interpolated onto the SOLPS grid correctly
        delta_step = 0.0001*np.min(np.abs(np.diff(rn)))
        
        # Remove any small negative diffusivities and throw a warning
        
        for i in range(len(rn)):
            if dn[i] < 0:
                print('dn[{}] = {:e}'.format(i,dn[i]))
                print('  Changed to dn[{}] = {:e}'.format(i,-dn[i]*1e-5))
                dn[i] = -dn[i] * 1e-5
            if ke[i] < 0:
                print('ke[{}] = {:e}'.format(i,ke[i]))
                print('  Changed to ke[{}] = {:e}'.format(i,-ke[i]*1e-2))
                ke[i] = -ke[i] * 1e-2
            if ki[i] < 0:
                print('ki[{}] = {:e}'.format(i,ki[i]))
                print('  Changed to ki[{}] = {:e}'.format(i,-ki[i]*1e-2))
                ki[i] = -ki[i] * 1e-2
        
        inlines = list()
        
        if solps5_0:
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
            
        else:
        
            inlines.append('&TRANSPORT\n')
            inlines.append('ndata( 1, 1, 1) = {} ,\n'.format(len(rn)))
            inlines.append("tdata(1, 1, 1, 1) = {:e} , tdata(2, 1, 1, 1) = {:e} ,\n".format(rn[0]-delta_step, dn[0]))
            for i in range(len(rn)-2):
                inlines.append("tdata(1, {}, 1, 1) = {:e} , tdata(2, {}, 1, 1) = {:e} ,\n".format(i+2,rn[i+1],i+2,dn[i+1]))
            inlines.append("tdata(1, {}, 1, 1) = {:e} , tdata(2, {}, 1, 1) = {:e} ,\n".format(len(rn), rn[-1]+delta_step, len(rn), dn[-1]))
            
    
            if carbon:
                inlines.append('ndata( 1, 1, 3) = {} ,\n'.format(len(rn)))
                inlines.append("tdata(1, 1, 1, 3) = {:e} , tdata(2, 1, 1, 3) = {:e} ,\n".format(rn[0]-delta_step, dn[0]))
                for i in range(len(rn)-2):
                    inlines.append("tdata(1, {}, 1, 3) = {:e} , tdata(2, {}, 1, 3) = {:e} ,\n".format(i+2, rn[i+1], i+2, dn[i+1]))
                inlines.append("tdata(1, {}, 1, 3) = {:e} , tdata(2, {}, 1, 3) = {:e} ,\n".format(len(rn), rn[-1]+delta_step, len(rn), dn[-1]))
                
                for j in range(4, 10):
                    inlines.append('ndata( 1, 1, {}) = {} ,\n'.format(j, len(rn)))
                    inlines.append("tdata(1, 1, 1, {}) = {:e} , tdata(2, 1, 1, {}) = {:e} ,\n".format(j, rn[0]-delta_step, j, dc[0]))
                    for i in range(len(rn)-2):
                        inlines.append("tdata(1, {}, 1, {}) = {:e} , tdata(2, {}, 1, {}) = {:e} ,\n".format(i+2, j, rn[i+1], i+2, j, dc[i+1]))
                    inlines.append("tdata(1, {}, 1, {}) = {:e} , tdata(2, {}, 1, {}) = {:e} ,\n".format(len(rn), j, rn[-1]+delta_step, len(rn),j,dc[-1]))
    
                for j in range(3, 10):
                    inlines.append('ndata( 1, 6, {}) = {} ,\n'.format(j, len(rn)))
                    inlines.append("tdata(1, 1, 6, {}) = {:e} , tdata(2, 1, 6, {}) = {:e} ,\n".format(j, rn[0]-delta_step, j, vrc[0]))
                    for i in range(len(rn)-2):
                        inlines.append("tdata(1, {}, 6, {}) = {:e} , tdata(2, {}, 6, {}) = {:e} ,\n".format(i+2, j, rn[i+1], i+2, j, vrc[i+1]))
                    inlines.append("tdata(1, {}, 6, {}) = {:e} , tdata(2, {}, 6, {}) = {:e} ,\n".format(len(rn), j, rn[-1]+delta_step, len(rn), j, vrc[-1]))
                    
            # Heat fluxes
    
            inlines.append('ndata( 1, 3, 1) = {} ,\n'.format(len(rn)))
            inlines.append("tdata(1, 1, 3, 1) = {:e} , tdata(2, 1, 3, 1) = {:e} ,\n".format(rn[0]-delta_step, ki[0]))
            for i in range(len(rn)-2):
                inlines.append("tdata(1, {}, 3, 1) = {:e} , tdata(2, {}, 3, 1) = {:e} ,\n".format(i+2, rn[i+1], i+2, ki[i+1]))
            inlines.append("tdata(1, {}, 3, 1) = {:e} , tdata(2, {}, 3, 1) = {:e} ,\n".format(len(rn), rn[-1]+delta_step, len(rn), ki[-1]))
    
            inlines.append('ndata( 1, 4, 1) = {} ,\n'.format(len(rn)))
            inlines.append("tdata(1, 1, 4, 1) = {:e} , tdata(2, 1, 4, 1) = {:e} ,\n".format(rn[0]-delta_step, ke[0]))
            for i in range(len(rn)-2):
                inlines.append("tdata(1, {}, 4, 1) = {:e} , tdata(2, {}, 4, 1) = {:e} ,\n".format(i+2, rn[i+1], i+2, ke[i+1]))
            inlines.append("tdata(1, {}, 4, 1) = {:e} , tdata(2, {}, 4, 1) = {:e} ,\n".format(len(rn), rn[-1]+delta_step, len(rn), ke[-1]))

            # if carbon:
            #     # Assign the same ion thermal diffusion coefficients (transport coefficient 3) from
            #     # species 1 to all other ion species
            #     # This seems to be incorrect, it breaks the reading
            #     for i in range(2, 10):
            #         inlines.append('addspec( {}, 3, 1) = {} ,\n'.format(i, i))
        
            
        inlines.append('no_pflux = .true.\n')
        inlines.append('/')
        
        # Write out file
        
        with open(inFile,'w') as f:
            for i in range(len(inlines)):
                f.write(inlines[i])

