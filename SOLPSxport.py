"""
This routine is used to generate new b2.transport.inputfile files for SOLPS
that will match the experimental upstream profiles.

This usually requires several iterations to match the experimental profiles well

The routine "main_mdsplus" runs the sequence in the correct order and returns
the SOLPSxport object for additional plotting and debugging if necessary

Once you've figured out all of the settings you need and are iterating on a run to
converge to the solution for transport coefficients, the routine "iterate_run" can
be useful to do this all quickly

R.S. Wilcox, J.M. Canik and J.D. Lore 2020
contact: wilcoxr@fusion.gat.com

Reference for this procedure:
https://doi.org/10.1016/j.jnucmat.2010.11.084
"""


import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from matplotlib.cm import get_cmap

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
        from SOLPSutils import readProf
        
        working = self.data['workdir']

        olddir = os.getcwd()
        os.chdir(working)
        if working[-1] != '/': working += '/'
        # Call 2d_profiles as default, so you don't accidentally look at old time steps
        if (not use_existing_last10) or (not os.path.isfile(working + 'ne3da.last10')):
            print("Calling '2d_profiles' in directory: " + working)
            os.system('2d_profiles')

        rx, ne_ = readProf('ne3da.last10')
        rx, dn_ = readProf('dn3da.last10')
        rx, te_ = readProf('te3da.last10')
        rx, ke_ = readProf('ke3da.last10')
        rx, ti_ = readProf('ti3da.last10')
        rx, ki_ = readProf('ki3da.last10')
        
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
        Either (1) provide the location of the saved profiles file
                   (should probably have extension *.pkl)
            or (2) give info to retrieve it from MDSplus
            **Second option requires access to atlas.gat.com
        """
        
        if profiles_file is None:
            from SOLPSutils import getProfDBPedFit
            
            if verbose: print('Getting profile fit data from MDSplus server on atlas.gat.com')
            
            self.timeid = timeid
            self.data['pedData']['fitVals'] = getProfDBPedFit(shotnum, timeid, runid)
            
        else:
            import pickle
            
            if verbose: print('Reading profile fit data from saved file: ' + profiles_file)
            
            with open(profiles_file, 'rb') as f:
                self.data['pedData']['fitVals'] = pickle.load(f)

    # ----------------------------------------------------------------------------------------
    
    def load_pfile(self, pfile_loc, plotit = False):
        from SOLPSutils import read_pfile
        self.timeid = pfile_loc[pfile_loc.rfind('.')+1:]

        pfile_dict = read_pfile(pfile_loc)

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
                        npsi=250, psiMax=1.05, plotit=False):
        """
        Get the fitted tanh profiles (need to have run loadProfDBPedFit already)
        """
    
        from SOLPSutils import calcTanhMulti
        
        psiProf = np.linspace(0, psiMax, npsi+1)
        
        self.data['pedData']['fitPsiProf'] = psiProf
        self.data['pedData']['fitProfs'] = {}
    
        # !!! Need to subtract shift from psiProf to move separatrix for power balance !!!

        # make ne profile
        if nemod == 'tnh0':
            necoef = self.data['pedData']['fitVals']['netnh0psi']['y']
            neprof = calcTanhMulti(necoef,psiProf)
        elif nemod == 'tanh':
            necoef = self.data['pedData']['fitVals']['netanhpsi']['y']
            neprof = calcTanhMulti(necoef,psiProf)
        self.data['pedData']['fitProfs']['neprof'] = neprof
            
        # make Te profile
        if temod == 'tnh0':
            tecoef = self.data['pedData']['fitVals']['tetnh0psi']['y']
            teprof = calcTanhMulti(tecoef,psiProf)
        elif nemod == 'tanh':
            tecoef = self.data['pedData']['fitVals']['tetanhpsi']['y']
            teprof = calcTanhMulti(tecoef,psiProf)
        self.data['pedData']['fitProfs']['teprof'] = teprof
        
        if ncmod == 'spl':
            zfzpsi = self.data['pedData']['fitVals']['zfz1splpsi']['x']
            zfzprof = self.data['pedData']['fitVals']['zfz1splpsi']['y']
            zfzfunc = interp1d(zfzpsi, zfzprof, bounds_error = False,
                               fill_value = zfzprof[np.argmax(zfzpsi)])
            # extrapolate to psin>1 using highest value of psin available
        
            self.data['pedData']['fitProfs']['ncprof'] = zfzfunc(psiProf) * neprof / 6
            
        elif ncmod=='tanh':
            nccoef = self.data['pedData']['fitVals']['nztanhpsi']['y']
            ncprof = calcTanhMulti(nccoef,psiProf)
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
            ax[1].set_xlim([0, psiMax+0.01])
            
            ax[0].set_title('Experimental Pedestal Fits')
            
            plt.show(block = False)

    # ----------------------------------------------------------------------------------------
        
    def getProfsOMFIT(self, prof_folder, prof_filename_prefix, min_npsi = 100,
                      psiMax=1.05, plotit = False):
        """
        Reads the prof files from OMFIT fits (saved as prof*.txt files) and produce fits
          These were made using 'OMFIT_tools.py'
          (they're just text files with individual fitted profiles)
        
        Only plots if the profiles are remapped
        """
        from SOLPSutils import readProf
        
        # Read in the prof*.txt files
        
        if os.path.isfile(os.path.join(prof_folder,prof_filename_prefix + '_T_D.txt')):
            profs = ['n_e', 'T_e', 'n_12C6', 'T_D']
        else:
            profs = ['n_e', 'T_e', 'n_12C6', 'T_12C6']
    
        psin = {}
        vals = {}
        for fit in profs:
            psin_read, vals_read = readProf(prof_filename_prefix + '_' + fit + '.txt',
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

        from SOLPSutils import loadg, B2pl
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

        dsa, crLowerLeft = B2pl('0 crx writ jxa f.y', wdir = wdir)
        # dummy, crLowerRight = B2pl('1 crx writ jxa f.y', wdir = wdir)
        # Only 2 unique psi values per cell, grab 0 and 2
        dummy, crUpperLeft = B2pl('2 crx writ jxa f.y', wdir = wdir)  # all x inds are the same
        dummy, czLowerLeft = B2pl('0 cry writ jxa f.y', wdir = wdir)
        dummy, czUpperLeft = B2pl('2 cry writ jxa f.y', wdir = wdir)
        ncells = len(dummy)

        g = loadg(self.data['gfile_loc'])
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
        from SOLPSutils import B2pl

        # x variable is identical for all of these
        x_fTot, fluxTot = B2pl("fnay za m* 0 0 sumz sy m/ writ jxa f.y")
        dummy, fluxConv = B2pl("na za m* vlay m* 0 0 sumz sy m/ writ jxa f.y")
        dummy, na = B2pl("na 0 0 sumz writ jxa f.y")
        dummy, hy1 = B2pl("hy1 writ jxa f.y")
        dummy, qe = B2pl("fhey sy m/ writ jxa f.y")
        dummy, qi = B2pl("fhiy sy m/ writ jxa f.y")
        

        for c in [fluxTot, fluxConv]:
            if not c:
                print("WARNING: Variable not populated by b2plot in getSOLPSfluxProfs")
                print("  Make sure ncl_ncar and netcdf modules are loaded")
                break

        self.data['solpsData']['profiles']['x_fTot'] = np.array(x_fTot)
        self.data['solpsData']['profiles']['fluxTot'] = np.array(fluxTot)
        self.data['solpsData']['profiles']['fluxConv'] = np.array(fluxConv)
        self.data['solpsData']['profiles']['na'] = np.array(na)
        self.data['solpsData']['profiles']['hy1'] = np.array(hy1)
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
        from SOLPSutils import B2pl

        x_nc, nc_solps = B2pl("na 8 zsel psy writ jxa f.y")
        dummy, flux_carbon = B2pl("fnay 8 zsel psy writ jxa f.y")  # x variables are the same
        dummy, vr_carbon = B2pl("vlay 8 zsel writ jxa f.y")
        
        for c in [flux_carbon, vr_carbon]:
            if not c:
                print("WARNING: Variable not populated by b2plot in getSOLPSCarbonProfs")
                print("  Make sure ncl_ncar and netcdf modules are loaded")
                break

        self.data['solpsData']['profiles']['x_nC'] = np.array(x_nc)
        self.data['solpsData']['profiles']['nC'] = np.array(nc_solps)
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
    
    def calcXportCoef(self, plotit = True, Dn_min = 0.002, chie_min = 0.01, chii_min = 0.01,
                      Dn_max = 10, chie_max = 200, chii_max = 200, vrc_mag=0.0,
                      rad_loc_for_exp_decay = 1.0, ti_decay_len = 0.015, reduce_Ti = False,
                      use_ratio_bc = True, debug_plots = False, verbose = False):
        """
        Calculates the transport coefficients to be written into b2.transport.inputfile
        
        Requires experimental profiles to have already been saved to self.data

        Inputs:
          reduce_Ti:    Use a saved array to get the ratio between T_C (measured) and T_i
                        This ratio was calculated from Shaun Haskey's T_D measurements
                        for 171558 @ 3200 ms
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
        
        fluxTot = self.data['solpsData']['profiles']['fluxTot']
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

        
        flux = fluxTot - fluxConv  # Conductive portion of the total flux
        dnew_flux = -flux / gnexp_solpslocs_dsa

        if use_ratio_bc:
            dnew_ratio[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
            dnew_flux[-1] = dold[-1] * neold[-1] / expden_dsa_func(dsa[-1])
        

        dnew_ratio[0] = dnew_ratio[1]  # gaurd cells
        dnew_flux[0] = dnew_flux[1]
        # dnew_ratio[-1] = dnew_ratio[-2]
        # dnew_flux[-1] = dnew_flux[-2]
        

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
        # kenew_ratio[-1] = kenew_ratio[-2]
        # kenew_flux[-1] = kenew_flux[-2]
        
        # Ti and ki
    
        tiexp = 1.0e3*self.data['pedData']['fitVals']['tisplpsi']['y']
        tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']
        
        if reduce_Ti:
            saved_ratio_file_loc = '/home/wilcoxr/profiles/171558/T_D_C_ratio.txt'
            print('Reducing T_D according to ratio of T_D / T_C from ' + saved_ratio_file_loc)

            with open(saved_ratio_file_loc, 'r') as f:
                lines = f.readlines()

            psin_ratio = []
            T_DC_ratio = []  # The ratio T_D / T_C from 171558
    
            for line in lines:
                elements = line.split()
                if elements[0] != '#':
                    psin_ratio.append(np.float(elements[0]))
                    T_DC_ratio.append(np.float(elements[1]))

            T_ratio_fit = np.interp(tiexppsi, np.array(psin_ratio), np.array(T_DC_ratio), left = 1)
            # if > given range, chooses endpoint
            ti_reduced = tiexp * T_ratio_fit
            
            if debug_plots:
                plt.figure()
                plt.plot(tiexppsi, tiexp/1e3, '--k', lw=2, label = 'T$_C$')
                plt.plot(tiexppsi, ti_reduced/1e3, 'r', lw=2, label = 'T$_D$ (inferred)')
                plt.xlabel('$\psi_n$')
                plt.ylabel('T$_i$ (keV)')
                plt.legend(loc='best')
                
            tiexp = ti_reduced
                
        
        # Modify Ti profile to decay exponentially outside separatrix
        timin = 1.0
        sep_ind = np.argmin(np.abs(tiexppsi - rad_loc_for_exp_decay))
        tiexp[sep_ind:] = (tiexp[sep_ind] - timin) * \
            np.exp(-(tiexppsi[sep_ind:]-tiexppsi[sep_ind]) / ti_decay_len) + timin
        
        
        dsa_tiprofile = psi_to_dsa_func(tiexppsi)
        
        gtiold = np.gradient(tiold) / np.gradient(dsa)
        gtiexp = np.gradient(tiexp) / np.gradient(dsa_tiprofile)
        

        gtiexp_dsafunc = interp1d(dsa_tiprofile, gtiexp, kind='linear', fill_value = 'extrapolate')
        # only very minor extrapolation should be required, like: 1.02 vs 1.02005657

        gtiexp_solpslocs = gtiexp_dsafunc(dsa)
        
        # Set boundary condition to get Ti[-1] right
        expTi_dsa_func = interp1d(dsa_tiprofile, tiexp, fill_value = 'extrapolate')
        gtiexp_solpslocs[-1] = -expTi_dsa_func(dsa[-1]) / ti_decay_len
        
        
        kinew_ratio = (gtiold / gtiexp_solpslocs) * kiold

        # gradient has to be in dsa to work
        kinew_flux = -(qi - 2.5 * fluxTot * tiold * eV) / (neold * eV * gtiexp_solpslocs)

        if use_ratio_bc:
            kinew_ratio[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])
            kinew_flux[-1] = kiold[-1] * tiold[-1] / expTi_dsa_func(dsa[-1])

        kinew_ratio[0] = kinew_ratio[1]   # gaurd cells
        kinew_flux[0] = kinew_flux[1]
        # kinew_ratio[-1] = kinew_ratio[-2]
        # kinew_flux[-1] = kinew_flux[-2]
        
        
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
            self.plotXportCoef(tiexp)

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

            plt.show(block=False)

    # ----------------------------------------------------------------------------------------

    def plotXportCoef(self, tiexp = None):
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
        tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']
        if tiexp is None:
            tiexp = self.data['pedData']['fitVals']['tisplpsi']['y']


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


        headroom = 1.05
        xlims = [np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01]

        f, ax = plt.subplots(2, 3, sharex = 'all')
        ax[0, 0].plot(psi_data_fit, neexp / 1.0e19, '--bo', lw = 1, label = 'TS data')
        ax[0, 0].plot(psi_solps, neold / 1.0e19, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0, 0].legend(loc = 'best',fontsize=14)
        ax[0, 0].set_ylim([0, max_ne*headroom])
        ax[0, 0].grid('on')

        ax[1, 0].plot(psi_solps, dnew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 0].plot(psi_solps, dnew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 0].plot(psi_solps, dold, '-xr', lw = 2, label = 'old')
        if coef_limits['Dn_min'] is not None:
            ax[1, 0].plot(xlims, [coef_limits['Dn_min'], coef_limits['Dn_min']], '--m')
        if coef_limits['Dn_max'] is not None:
            ax[1, 0].plot(xlims, [coef_limits['Dn_max'], coef_limits['Dn_max']], '--m')
        ax[1, 0].set_ylabel('D (m$^2$/s)')
        ax[1, 0].set_xlabel('$\psi_N$')
        ax[1, 0].set_ylim([0, max_dn*headroom])
        ax[1, 0].grid('on')

        ax[0, 1].plot(psi_data_fit, teexp / 1.0e3, '--bo', lw = 1, label = 'Data')
        ax[0, 1].plot(psi_solps, teold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 1].set_ylabel('T$_e$ (keV)')
        ax[0, 1].set_ylim([0, max_Te*headroom])
        ax[0, 1].grid('on')
        if self.data['workdir_short'] is not None:
            ax[0, 1].set_title(self.data['workdir_short'], fontsize=10)

        ax[1, 1].plot(psi_solps, kenew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 1].plot(psi_solps, kenew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 1].plot(psi_solps, keold, '-xr', lw = 2, label = 'old')
        if coef_limits['chie_min'] is not None:
            ax[1, 1].plot(xlims, [coef_limits['chie_min'], coef_limits['chie_min']], '--m')
        if coef_limits['chie_max'] is not None:
            ax[1, 1].plot(xlims, [coef_limits['chie_max'], coef_limits['chie_max']], '--m')
        ax[1, 1].set_ylabel('$\chi_e$ (m$^2$/s)')
        ax[1, 1].set_xlabel('$\psi_N$')
        ax[1, 1].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        ax[1, 1].set_ylim([0, max_ke*headroom])
        ax[1, 1].grid('on')

        ax[0, 2].plot(psi_solps, tiold / 1.0e3, 'xr', lw = 2, label = 'SOLPS')
        ax[0, 2].plot(tiexppsi, tiexp / 1.0e3, '--bo', lw = 1, label = 'Data')
        ax[0, 2].set_ylabel('T$_i$ (keV)')
        ax[0, 2].set_ylim([0, max_Ti*headroom])
        ax[0, 2].grid('on')

        ax[1, 2].plot(psi_solps, kinew_flux, '-ok', lw = 2, label = 'updated (fluxes)')
        ax[1, 2].plot(psi_solps, kinew_ratio, '-+c', lw = 1, label = 'updated (gradients)')
        ax[1, 2].plot(psi_solps, kiold, '-xr', lw = 2, label = 'old')
        if coef_limits['chii_min'] is not None:
            ax[1, 2].plot(xlims, [coef_limits['chii_min'], coef_limits['chii_min']], '--m')
        if coef_limits['chii_max'] is not None:
            ax[1, 2].plot(xlims, [coef_limits['chii_max'], coef_limits['chii_max']], '--m')
        ax[1, 2].set_ylabel('$\chi_i$ (m$^2$/s)')
        ax[1, 2].set_xlabel('$\psi_N$')
        ax[1, 2].set_xlim(xlims)
        ax[1, 2].set_ylim([0, max_ki*headroom])
        ax[1, 2].grid('on')
        ax[1, 2].legend(loc='best', fontsize=12)

        ax[0, 0].set_xticks(np.arange(0.84, 1.05, 0.04))
        ax[0, 0].set_xlim(xlims)
        plt.tight_layout()

        plt.show(block = False)

    # ----------------------------------------------------------------------------------------
    
    def plot_profiles(self):
        """
        Plot the upstream profiles from SOLPS compared to the experiment
        """
        # if 'xportCoef' not in self.data['solpsData']:
        #     print('Transport coefficients not yet calculated!! Calculating them using defaults')
        #     self.calcXportCoef(plotit = False,debug_plots = False)

        headroom = 1.04
        
        # Load SOLPS profiles and transport coefficients

        psi_solps = self.data['solpsData']['psiSOLPS']
        neold = self.data['solpsData']['last10']['ne']
        dold = self.data['solpsData']['last10']['dn']
        teold = self.data['solpsData']['last10']['te']
        keold = self.data['solpsData']['last10']['ke']
        tiold = self.data['solpsData']['last10']['ti']
        kiold = self.data['solpsData']['last10']['ki']
        
        # Load experimental profiles

        psi_data_fit = self.data['pedData']['fitPsiProf']
        neexp = 1.0e20 * self.data['pedData']['fitProfs']['neprof']
        teexp = 1.0e3*self.data['pedData']['fitProfs']['teprof']
        tiexp = 1.0e3*self.data['pedData']['fitVals']['tisplpsi']['y']
        tiexppsi = self.data['pedData']['fitVals']['tisplpsi']['x']


        dnew_ratio = self.data['solpsData']['xportCoef']['dnew_ratio']
        kenew_ratio = self.data['solpsData']['xportCoef']['kenew_ratio']
        kinew = self.data['solpsData']['xportCoef']['kinew']


        # Find limits of Te, Ti for plots
        TS_inds_in_range = np.where(psi_data_fit > np.min(psi_solps))[0]
        Ti_inds_in_range = np.where(tiexppsi > np.min(psi_solps))[0]
        max_ne = np.max([np.max(neold), np.max(neexp[TS_inds_in_range])]) / 1.0e19
        max_Te = np.max([np.max(teold), np.max(teexp[TS_inds_in_range])])
        max_Ti = np.max([np.max(tiold), np.max(tiexp[Ti_inds_in_range])])


        f, ax = plt.subplots(2, sharex = 'all')
        ax[0].plot(psi_data_fit, neexp / 1.0e19, '--bo', lw = 1, label = 'Experimental Data')
        ax[0].plot(psi_solps, neold / 1.0e19, 'xr', lw = 2, mew=2, ms=8, label = 'SOLPS')
        ax[0].set_ylabel('n$_e$ (10$^{19}$ m$^{-3}$)')
        ax[0].legend(loc = 'best')
        ax[0].set_ylim([0, max_ne * headroom])
        ax[0].grid('on')

        # ax[1, 0].plot(psi_solps, dold, '-xr', lw = 2)
        # ax[1, 0].plot(psi_solps, dnew_ratio, '-ok', lw = 2, label = 'Data')
        # ax[1, 0].set_ylabel('D')
        # ax[1, 0].set_xlabel('$\psi_N$')
        # ax[1, 0].grid('on')

        ax[1].plot(psi_data_fit, teexp, '--bo', lw = 1, label = 'Experimental Data')
        ax[1].plot(psi_solps, teold, 'xr', lw = 2, mew=2, ms=8, label = 'SOLPS')
        ax[1].set_ylabel('T$_e$ (eV)')
        ax[1].set_ylim([0, max_Te * headroom])
        ax[1].set_yticks(np.arange(0, max_Te * headroom + 200, 200))
        ax[1].grid('on')
        ax[1].set_xlabel('$\psi_N$')

        # ax[1, 1].plot(psi_solps, keold, '-xr', lw = 2)
        # ax[1, 1].plot(psi_solps, kenew_ratio, '-ok', lw = 2, label = 'Data')
        # ax[1, 1].set_ylabel('$\chi_e$')
        # ax[1, 1].set_xlabel('$\psi_N$')
        # ax[1, 1].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        # ax[1, 1].grid('on')

        # ax[0, 2].plot(psi_solps, tiold, 'xr', lw = 2, label = 'SOLPS')
        # ax[0, 2].plot(tiexppsi, tiexp, '--bo', lw = 1, label = 'Data')
        # ax[0, 2].set_ylabel('T$_i$ (eV)')
        # ax[0, 2].set_ylim([0, max_Ti * headroom])
        # ax[0, 2].grid('on')

        # ax[1, 2].plot(psi_solps, kiold, '-xr', lw = 2)
        # ax[1, 2].plot(psi_solps, kinew, '-ok', lw = 2, label = 'Data')
        # ax[1, 2].set_ylabel('$\chi_i$')
        # ax[1, 2].set_xlabel('$\psi_N$')
        # ax[1, 2].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        # ax[1, 2].grid('on')

        ax[0].set_xticks(np.arange(0.84, 1.05, 0.04))
        ax[0].set_xlim([np.min(psi_solps) - 0.01, np.max(psi_solps) + 0.01])
        plt.tight_layout()

        plt.show(block = False)
    
    # ----------------------------------------------------------------------------------------
    
    def writeXport(self, new_filename = 'new.transport.inputfile', solps5_0 = False):
        """
        Write the b2.transport.inputfile using values saved in this object
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
        dn = self.data['solpsData']['xportCoef']['dnew_flux']
        ke = self.data['solpsData']['xportCoef']['kenew_flux']
        ki = self.data['solpsData']['xportCoef']['kinew_flux']
        vrc = self.data['solpsData']['xportCoef']['vr_carbon']
        dc = self.data['solpsData']['xportCoef']['D_carbon']
        carbon = self.data['carbon']
        
        # Step the boundary points out a tiny bit so that they are
        # interpolated onto the SOLPS grid correctly
        delta_step = 0.5*np.min(np.abs(np.diff(rn)))
        
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
        
            
        inlines.append('no_pflux = .true.\n')
        inlines.append('/')
        
        # Write out file
        
        with open(inFile,'w') as f:
            for i in range(len(inlines)):
                f.write(inlines[i])

# ----------------------------------------------------------------------------------------


def main_omfit(topdir, subfolder, gfile_loc, prof_folder = None,
               prof_filename_prefix = 'prof171558_3200',
               new_filename = 'test.transport.inputfile',
               use_existing_last10 = False,
               carbon = True, plotall = False, debug_plots = False, plot_xport_coeffs = True):
    """
    **This has not yet been fixed to work with the current version of these codes**
    """
    print("WARNING: This routine is likely to break")
    print("         Updates need to be made before it works with OMFIT")

    print("Initializing SOLPSxport")
    xp = SOLPSxport(workdir = topdir + subfolder, gfile_loc = gfile_loc, carbon_bool = carbon)
    print("Running getSOLPSlast10Profs")
    xp.getSOLPSlast10Profs(plotit = plotall, use_existing_last10 = use_existing_last10)
    print("Running getProfsOMFIT")
    xp.getProfsOMFIT(prof_folder = prof_folder, prof_filename_prefix = prof_filename_prefix,
                     min_npsi = 100, psiMax = 1.05, plotit = plotall)
    print("Running calcPsiVals")
    xp.calcPsiVals(plotit = plotall)

    if carbon:
        print("Running getSOLPSCarbonProfs")
        xp.getSOLPSCarbonProfs(plotit = plotall)

    print("Running calcXportCoeff")
    xp.calcXportCoef(plotit = plotall or plot_xport_coeffs, debug_plots = debug_plots)

    print("Running writeXport")
    xp.writeXport(new_filename = new_filename)

    return xp

# ----------------------------------------------------------------------------------------


def main_mdsplus(rundir, gfile_loc, new_filename = 'new.transport.inputfile',
                 profiles_fileloc = None, shotnum = None, ptime = '1550', prunid = 'm8099',
                 use_existing_last10 = False, reduce_Ti = False, carbon = True,
                 plotall = False, plot_xport_coeffs = True, Dn_min = 0.002, vrc_mag = 0.0,
                 ti_decay_len = 0.015, verbose = False):
    """
    Driver for the code using Osborne profile fits saved in MDSplus
    """
    
    if shotnum is None: shotnum = int(gfile_loc[-12:-6])
    if ptime is None: ptime = int(gfile_loc[-4:])
    
    print("Initializing SOLPSxport")
    xp = SOLPSxport(workdir = rundir, gfile_loc = gfile_loc, carbon_bool = carbon)
    print("Running calcPsiVals")
    xp.calcPsiVals(plotit = plotall)
    print("Running getSOLPSlast10Profs")
    xp.getSOLPSlast10Profs(plotit = plotall, use_existing_last10 = use_existing_last10)
    xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptime, prunid, verbose = True)
    print("Populating PedFits")
    xp.populatePedFits(nemod='tanh',temod='tanh',ncmod='spl',npsi=250,psiMax=1.05, plotit=plotall)
    print("Getting flux profiles")
    xp.getSOLPSfluxProfs(plotit = plotall)

    if carbon:
        print("Running getSOLPSCarbonProfs")
        xp.getSOLPSCarbonProfs(plotit = plotall)

    print("Running calcXportCoeff")
    xp.calcXportCoef(plotit = plotall or plot_xport_coeffs, reduce_Ti = reduce_Ti, Dn_min = Dn_min,
                     ti_decay_len = ti_decay_len, vrc_mag = vrc_mag, verbose = verbose)

    print("Running writeXport")
    xp.writeXport(new_filename = new_filename)
    
    return xp

# ----------------------------------------------------------------------------------------


def track_inputfile_iterations(rundir = None, carbon = True, cmap = 'viridis'):
    """
    Track the evolution of the b2.transport.inputfile transport
    coefficients through and evolving transport matching job
    
    Append b2.transport.inputfile filenames with numbers and leave them
    in the same run directory
    
    Requires everything to be on the same grid with the same species
    """
    from SOLPSutils import read_b2_transport_inputfile

    if rundir is None:
        print("No run directory given, selecting cwd:")
        rundir = os.getcwd()
        print(rundir)

    cm = get_cmap(cmap)
    
    _, _, filenames = next(os.walk(rundir), (None, None, []))
    inputfile_list = [fn for fn in filenames if
                      (fn[:22] == 'b2.transport.inputfile' and fn[-1] != '~')]
    inputfile_list.sort()
    if len(inputfile_list[0]) == 22:  # most recent, unnumbered inputfile should be at the end
        inputfile_list = inputfile_list[1:] + [inputfile_list[0]]
        
    ninfiles = len(inputfile_list)
    
    if rundir[-1] != '/': rundir += '/'
    

    dn_sep = np.zeros(ninfiles)
    ki_sep = np.zeros(ninfiles)
    ke_sep = np.zeros(ninfiles)
    dn_bdy = np.zeros(ninfiles)
    ki_bdy = np.zeros(ninfiles)
    ke_bdy = np.zeros(ninfiles)
    
    f, ax = plt.subplots(3, sharex = 'all')
        
    for i in range(ninfiles):
        
        infile = read_b2_transport_inputfile(rundir + inputfile_list[i], carbon = carbon)
        
        sep_ind = np.argmin(np.abs(infile['rn']))
        dn_sep[i] = infile['dn'][sep_ind]
        ki_sep[i] = infile['ki'][sep_ind]
        ke_sep[i] = infile['ke'][sep_ind]
        dn_bdy[i] = infile['dn'][-1]
        ki_bdy[i] = infile['ki'][-1]
        ke_bdy[i] = infile['ke'][-1]

        ax[0].plot(infile['rn'], infile['dn'], '-x', color = cm(i / (float(ninfiles) - 1)),
                   label = inputfile_list[i][13:])
        ax[1].plot(infile['rn'], infile['ki'], '-x', color = cm(i / (float(ninfiles) - 1)))
        ax[2].plot(infile['rn'], infile['ke'], '-x', color = cm(i / (float(ninfiles) - 1)))

    ax[0].set_ylabel('dn')
    ax[1].set_ylabel('ki')
    ax[2].set_ylabel('ke')
    ax[-1].set_xlabel('rn')
    ax[0].legend(loc='best')
    for j in range(len(ax)):
        ax[j].grid('on')
        
    plt.figure()
    plt.plot(range(ninfiles), dn_sep, '-xk', lw=2, label = 'Dn')
    plt.plot(range(ninfiles), ki_sep, '-ob', lw=2, label = 'ki')
    plt.plot(range(ninfiles), ke_sep, '-or', lw=2, label = 'ke')
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.title('Transport coefficient evolution at separatrix')
    plt.grid('on')

    plt.figure()
    plt.plot(range(ninfiles), dn_bdy, '-xk', lw=2, label = 'Dn')
    plt.plot(range(ninfiles), ki_bdy, '-ob', lw=2, label = 'ki')
    plt.plot(range(ninfiles), ke_bdy, '-or', lw=2, label = 'ke')
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.title('Transport coefficient evolution at boundary')
    plt.grid('on')
    
    plt.show(block=False)

# ----------------------------------------------------------------------------------------


def increment_run(rundir, gfile_loc, new_filename = 'new.transport.inputfile',
                  profiles_fileloc = None, shotnum = None, ptime = None, prunid = None,
                  use_existing_last10 = False, reduce_Ti = False,
                  carbon = True, plotall = False, plot_xport_coeffs = True,
                  ntim_new = 100, dtim_new = '1.0e-6', Dn_min = 0.01):
    """
    This routine runs the main calculation of transport coefficients, then saves
    the old b2.transport.inputfile and b2fstati files with the iteration number
    and updates the b2mn.dat file with short time steps in preparation for the new run
    """

    olddir = os.getcwd()
    os.chdir(rundir)
    
    xp = main_mdsplus(rundir = rundir, gfile_loc = gfile_loc, new_filename = new_filename,
                      profiles_fileloc = profiles_fileloc, shotnum = shotnum, ptime = ptime,
                      prunid = prunid, Dn_min = Dn_min, use_existing_last10 = use_existing_last10,
                      reduce_Ti = reduce_Ti, carbon = carbon, plotall = plotall,
                      plot_xport_coeffs = plot_xport_coeffs, verbose=False)
    
    allfiles = os.listdir('.')
    all_incs = [int(i[22:]) for i in allfiles if i[:22]=='b2.transport.inputfile' and i[-1]!='~' and i[-1]!='e']
    
    inc_num = np.max(all_incs)
    os.rename('b2fstati', 'b2fstati' + str(inc_num+1))
    os.rename('b2.transport.inputfile', 'b2.transport.inputfile' + str(inc_num+1))
    os.rename(new_filename, 'b2.transport.inputfile')
    # os.remove('run.log')
    for filename in allfiles:
        if filename[-7:]=='.last10':
            os.remove(filename)
    # os.remove('*.last10')
    # os.system('rm *.last10')  Doesn't work for some reason
    
    print('modifying b2mn.dat')

    with open('b2mn.dat', 'r') as f:
        lines = f.readlines()
        
    # os.rename('b2mn.dat','b2mn.dat_old')  (this looks good, so just overwrite it from now on)
    
    for i, line in enumerate(lines):
        if line[:13] == "'b2mndr_ntim'":
            lines[i] = lines[i][:35] + "'" + str(ntim_new) + "'\r\n"
            
        if line[:13] == "'b2mndr_dtim'":
            lines[i] = lines[i][:35] + "'" + str(dtim_new) + "'\r\n"
            break

    with open('b2mn.dat', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])
            
    os.chdir(olddir)
