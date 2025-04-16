"""
This driver is used to generate new b2.transport.inputfile and
b2.transport.parameters files for SOLPS that will come closer to
matching the experimental upstream profiles. If things go well, after
several iterations, this will produce radial ne, Te and Ti profiles in
SOLPS at the outboard midplane that match the experimental profiles
provided. It does this through the generation of an object of class 'SOLPSxport'

The routine "main" runs the sequence in the correct order and returns
the SOLPSxport object for additional plotting and debugging if necessary

Note: This algorithm attempts to match the SOLPS profile gradients to the
experimental gradients, but the absolute values will not necessarily
match if core flux boundary conditions are used (which they should be!)
and the experimental profiles do not diminish to zero at the grid boundary.

**So far this is only tested for cases with D or D+C, cases with other impurities may not work yet**


Instructions for command line call:

-> (if python is not already available) -> $ module load python
-> Source SOLPS-ITER setup file for b2plot calls
-> Navigate to run directory
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -s <shot number> -t <profile time ID> -r <profile run ID>
or if you already have a saved profiles file (.pkl):
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -p <profiles file location>
optional flags: --chii_eq_chie (or) --ti_eq_te


For calling within interactive Python session:

-> Source SOLPS-ITER setup file for b2plot and 2d_profiles calls
   (this step is no longer necessary if you have already run 2d_profiles to produce .last10 files)
-> Make sure SOLPSxport_dr is located somewhere in the $PYTHONPATH environment variable
-> Navigate to run directory with b2fstate and other run files
$ ipython
In [1]: import SOLPSxport_dr as sxdr  # (depending on directory structure, might need to be SOLPSxport.SOLPSxport_dr)
In [2]: xp = sxdr.main(gfile_loc = '../g123456.02000', profiles_fileloc = 'prof_123456_2000.pkl')


Requirements:
-Code depends on the SOLPSxport class contained in SOLPSxport.py, as well as routines in SOLPSutils.py
-An existing SOLPS run, which will be used to estimate the next iteration's
radial transport coefficients
-Experimental Te, ne and Ti profiles, preferably saved in the MDSplus database if DIII-D ("Tom's tools")
-.last10 files generated by running '2d_profiles' in the run directory, which uses data from the
  final 10 time steps written in run.log. If you do not normally write output to run.log, you'll
  need to launch an additional run of at least 10 time steps that writes to the run.log file

-(only if you want these routines to be able to call B2plot and 2d_profiles) Source setup.csh from
  a SOLPS-ITER distribution that can run b2plot before launching ipython and importing this module

This usually requires several iterations to match the experimental profiles well
(at least ~5 if starting from flat transport coefficients)

Once you've figured out all the settings you need and are iterating on a run to
converge to the solution for transport coefficients, the routine "increment_run" can
be useful to do this all quickly

R.S. Wilcox, J.M. Canik and J.D. Lore 2020-2025
contact: wilcoxrs@ornl.gov

Reference for this procedure:
https://doi.org/10.1016/j.jnucmat.2010.11.084
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from scipy import interpolate

import SOLPSutils as sut
import SOLPSxport as sxp
import inspect
try:
    import json
except:
    print('json module not available, some functionality not available')

plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.facecolor':'w'})
plt.rcParams.update({'mathtext.default': 'regular'})

# ----------------------------------------------------------------------------------------


def main(gfile_loc = None, new_filename='b2.transport.inputfile_new',
         profiles_fileloc=None, shotnum=None, ptimeid=None, prunid=None,
         nefit='tanh', tefit='tanh', ncfit='spl', chii_eq_chie = False, ti_eq_te = False,
         Dn_min=0.001, vrc_mag=0.0, Dn_max=200, use_ratio_bc = True, update_d_only=False,
         chie_use_grad = False, chii_use_grad = False, new_b2xportparams = True,
         chie_min = 0.01, chii_min = 0.01, chie_max = 400, chii_max = 400,
         reduce_Ti_fileloc = None, update_old_last10s = False, reject_fewer_than_10=True,
         fractional_change = 1, elec_prof_rad_shift = 0, ti_fileloc = None,
         impurity_list = ['c'], use_existing_last10= False, plot_xport_coeffs=True,
         plotall=False, plot_fluxes = False, verbose=False, figblock=False,
         rad_loc_for_exp_decay=1.0, ti_decay_len=0.015, te_decay_len = None, ne_decay_len = None,
         ti_decay_min=1, te_decay_min = 1, ne_decay_min = 1e18):
    """
    Driver for the code, returns an object of class 'SOLPSxport'

    Inputs:
      rundir            Location of SOLPS run directory
                        --> depricated, assume you are in the directory (this is required for b2plot calls)
      gfile_loc         Location of gfile used for SOLPS grid (for mapping SOLPS grid to psin)
      profiles_fileloc  (optional) Location of a .pkl file with saved Tom's tools profile
                        fit coefficients (produced by getProfDBPedFit from SOLPSutils)
                        OR if not a .pkl file, can use a pfile as long as profiles extend far enough into SOL
      shotnum,ptimeid,prunid  Profile identifying shot number, timeid and runid (from Tom's tools)
                        (this is uneccessary if you have a .pkl file given in profiles_fileloc)
      xxfit             Fit function used in each profile fit (xx => ne, te and nc)
      chii_eq_chie      Set to True to ignore Ti profile and just set chi_i = chi_e
      ti_eq_te          Set to True to set Ti = Te (will override any other modifications to Ti or chi_i)
      Dn_min            Set a floor for the allowable particle diffusion coefficient
      Dn_max            Set a maximum for the allowable particle diffusion coefficient
      chie_max          Set a maximum for the allowable electron energy diffusion coefficient
      update_d_only     Do not update chi values (for early iterations when particle transport is way off)
      use_ratio_bc      Set to True if you want to modify the last point on the grid to attempt to enforce
                        a boundary condition (this seems to improve convergence to experimental profiles
                        a bit for narrow grids, but ideally would do this using BCCON, BCENE and BCENI
                        in b2.boundary.parameters)
      vrc_mag           Hard-coded carbon impurity pinch, for trying to match nC profiles
                        (leave zero unless you also change the function within calcXportCoeffs)
      rad_loc_for_exp_decay: Radial position in psin for the beginning of exponential decay
      ti_decay_len      Decay length (at the outboard midplane) for imposed exponential falloff
                        for experimental Ti, beginning at rad_loc_for_exp_decay (default = separatrix)
                        (this is useful because impurity CER is incorrect in SOL)
      te_decay_len      "" for Te
      ne_decay_len      "" for ne
      ti_decay_min      far-SOL Ti to decay to (eV)
      te_decay_min      far-SOL Te to decay to (eV)
      ne_decay_min      far-SOL ne to decay to (m^-3)
      chie/i_use_grad   Use ratio of the gradients for new values of chi_e/i, rather than fluxes
      new_b2xportparams Produces updated b2.transport.parameters so that D, X are set in PFR to match first
                        radial cell of SOL (default is on)
      use_existing_last10  Set to True if you have already run 2d_profiles to produce .last10 files
                           in the run folder to save time. Otherwise this will call 2d_profiles so
                           that you don't accidentally use .last10 files from a previous iteration
                           with out-of-date SOLPS profiles
      reduce_Ti_fileloc Set to None to use T_D = T_C from MDS+ profile fit
                        *On GA clusters (Iris and Omega), example file is located here:
                        '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt'
      update_old_last10s  Set to True to copy the last10 files to last10.old for comparison with the next iteration
      reject_fewer_than_10  Stops everything if fewer than 10 time steps were run on most recent SOLPS call
                            (efault = True)
      fractional_change Set to number smaller than 1 if the incremental change is too large and
                        you want to take a smaller step
      elec_prof_rad_shift: Apply a radial shift to experimental electron profiles
                        (in units of psi_n, positive shifts profiles outward so separatrix is hotter)
      ti_fileloc        Separate file with Ti data (overwrites previous fits)
      impurity_list     List of all the impurities included in the plasma simulation
                        (not tested yet for anything other than 'c')
      plot_xport_coeffs Plot the SOLPS and experimental profiles, along with the previous
                        and next iteration of transport coefficients
      plotall           Produce a bunch more plots from the subroutines used in the process
      plot_fluxes       Plot the poloidally-integrated radial fluxes for particles and energy
      verbose           Set to True to print a lot more outputs to terminal
      figblock          Set to True if calling from command line and you want to see figures
    Returns:
      Object of class 'SOLPSxport', which can then be used to plot, recall, or modify the saved data
      and rewrite a new b2.transport.inputfile
    """
    if 'json' in sys.modules:
        # Write dict of last call arguments as json file
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        argvals = {}
        for arg in args:
            argvals[arg] = values[arg]
        json.dump(argvals,open("SOLPSxport_args.last",'w'))

    if shotnum is None:
        try:
            shotnum = int(gfile_loc[gfile_loc.rfind('g')+1:gfile_loc.rfind('.')])
        except ValueError:
            if verbose: print("Can't determine shot number, setting to 0")
            shotnum = 0
    if ptimeid is None:
        try:
            ptimeid = int(gfile_loc[gfile_loc.rfind('.')+1:])
        except ValueError:
            if verbose: print("Can't determine time slice, setting to 0")
            ptimeid = 0
    ptimeid = int(ptimeid)
    shotnum = int(shotnum)

    print("Initializing SOLPSxport")
    xp = sxp.SOLPSxport(workdir=os.getcwd(), gfile_loc=gfile_loc, impurity_list=impurity_list)

    print("Reading SOLPS output")
    try:
        dsa = sut.read_dsa("dsa")
        b2mn = sut.scrape_b2mn("b2mn.dat")        
        geo = sut.read_b2fgmtry("../baserun/b2fgmtry")
        state = sut.read_b2fstate("b2fstate")
        xport = sut.read_transport_files(".", dsa=dsa, geo=geo, state=state)
    except:
        print('Failed to read output directly, will try using b2plot')
        sut.set_b2plot_dev(verbose=verbose)
        xp.b2plot_ready = True
        dsa = None
        b2mn = None
        geo = None
        state = None
        xport = None
        
    print("Running calcPsiVals")
    try:
        xp.calcPsiVals(plotit=plotall,geo=geo,b2mn=b2mn,dsa=dsa)
    except Exception as err:
        print('Exiting from SOLPSxport_dr\n')
        sys.exit(err)
    if use_existing_last10:
        print("Running getSOLPSlast10Profs")
        xp.getSOLPSlast10Profs(plotit=plotall, use_existing_last10=use_existing_last10)
    else:
        print("Getting OMP profiles from last 10 time steps in b2time")
        xp.getlast10profs_b2time(reject_fewer_than_10=reject_fewer_than_10, plotit=plotall)
        xp.write_last10files()
    # xp.getProfsOMFIT(prof_folder = prof_folder, prof_filename_prefix = prof_filename_prefix,
    #                  min_npsi = 100, psiMax = 1.05, plotit = plotall)
    if profiles_fileloc is None:  # try to call MDSplus only if no profiles_fileloc is given
        if prunid is not None:
            xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptimeid, prunid, verbose=True)
            print("Populating PedFits from MDSplus database")
            xp.populatePedFits(nemod=nefit, temod=tefit, ncmod=ncfit, npsi=250, plotit=plotall)
        else:
            print("ERROR: Need to provide eiter a profiles fileloc or MDSplus profile runid")
            sys.exit()
    elif profiles_fileloc[-4:] == '.pkl':
        xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptimeid, prunid, verbose=True)
        print("Populating PedFits")
        xp.populatePedFits(nemod=nefit, temod=tefit, ncmod=ncfit, npsi=250, plotit=plotall)
    elif profiles_fileloc[-4:] == '.dat':
        xp.readMastData(profiles_fileloc)
    else:
        print("Loading profiles from pfile")
        xp.load_pfile(profiles_fileloc, plotit=plotall)

    if ti_fileloc:
        xp.load_ti(ti_fileloc=ti_fileloc, verbose=True)

    if ti_eq_te:
        print("** Forcing target Ti = Te **")
        xp.data['expData']['fitProfs']['tipsi'] = xp.data['expData']['fitPsiProf']
        xp.data['expData']['fitProfs']['tiprof'] = xp.data['expData']['fitProfs']['teprof']

    print("Getting upstream flux profiles from poloidal index jxa")
    xp.getjxafluxprofs(plotit=plotall,dsa=dsa,b2mn=b2mn,geo=geo,state=state,xport=xport)

    if impurity_list:
        print("Running getSOLPSCarbonProfs")
        xp.getSOLPSCarbonProfs(plotit=plotall,dsa=dsa,b2mn=b2mn,geo=geo,state=state,xport=xport)

    print("Running calcXportCoeff")
    xp.calcXportCoef(plotit=plotall or plot_xport_coeffs, reduce_Ti_fileloc=reduce_Ti_fileloc, Dn_min=Dn_min,
                     vrc_mag=vrc_mag, verbose=verbose, Dn_max=Dn_max, update_d_only=update_d_only,
                     fractional_change=fractional_change, elec_prof_rad_shift=elec_prof_rad_shift,
                     chii_min=chii_min, chii_max=chii_max, chie_min=chie_min, chie_max=chie_max,
                     chii_eq_chie=chii_eq_chie, figblock=figblock, use_ratio_bc=use_ratio_bc,
                     ti_decay_len=ti_decay_len, te_decay_len=te_decay_len, ne_decay_len=ne_decay_len,
                     ti_decay_min=ti_decay_min, te_decay_min=te_decay_min, ne_decay_min=ne_decay_min,
                     rad_loc_for_exp_decay=rad_loc_for_exp_decay,
                     plot_gradient_method=(chii_use_grad or chie_use_grad))

    print("Writing to: " + new_filename)
    xp.writeXport(new_filename=new_filename, chie_use_grad=chie_use_grad, chii_use_grad=chii_use_grad,
                  chii_eq_chie=chii_eq_chie, update_d_only=update_d_only)

    psin_solps = list(xp.data['solpsData']['psiSOLPS'])

    # Modify b2.transport.parameters so that PFR has same transport coefficients as first cell of SOL
    if new_b2xportparams:
        first_sol_ind = psin_solps.index(min([i for i in psin_solps if i > 1]))
        dperp_pfr = xp.data['solpsData']['xportCoef']['dnew_flux'][first_sol_ind]

        if chie_use_grad:
            chieperp_pfr = xp.data['solpsData']['xportCoef']['kenew_ratio'][first_sol_ind]
        else:
            chieperp_pfr = xp.data['solpsData']['xportCoef']['kenew_flux'][first_sol_ind]

        if chii_eq_chie:
            chiiperp_pfr = chieperp_pfr
        else:
            if chie_use_grad:
                chiiperp_pfr = xp.data['solpsData']['xportCoef']['kinew_ratio'][first_sol_ind]
            else:
                chiiperp_pfr = xp.data['solpsData']['xportCoef']['kinew_flux'][first_sol_ind]

        sut.new_b2xportparams(fileloc='./b2.transport.parameters',
                              dperp=dperp_pfr, chieperp=chieperp_pfr, chiiperp=chiiperp_pfr, verbose=True)


    print("Checking particle and energy fluxes through the North boundary")
    xp.get_integrated_radial_fluxes(plotit=plot_fluxes, dsa=dsa, geo=geo, state=state, figblock=figblock)

    # gammaD = xp.data['solpsData']['profiles']['gammaD']
    # Qtot = xp.data['solpsData']['profiles']['Qtot']
    #
    # if gammaD[0] > 1e19:
    #     print(' Core D+ particle flux escaping domain radially: {:4.1f}%'.format(100 * gammaD[-1] / gammaD[0]))
    # else:
    #     print('assuming same number of radial cells in core and sol')
    #     print(' Sep. D+ particle flux escaping domain radially: {:4.1f}%'.format(100 * gammaD[-1] / gammaD[len(gammaD) / 2]))

    if 'gammaD' in xp.data['solpsData']['profiles'].keys():
        gammaD = xp.data['solpsData']['profiles']['gammaD']

        # If the core D+ particle flux is less than 5% of the peak, then the radially escaping flux is compared to
        # the separatrix particle flux rather than the core flux

        if gammaD[0] > 0.05*np.max(gammaD):
            print("\nInjected D particles escaping radial grid boundary: {:4.1f}%".format(100*gammaD[-1] / gammaD[0]))
        else:
            print('\nSep. D particle flux escaping radial grid boundary: {:4.1f}%'.format(100*gammaD[-1]/gammaD[len(psin_solps)/2]))

        print("Injected energy flux escaping radial grid boundary: {:4.1f}%\n".
              format(100*xp.data['solpsData']['profiles']['Qtot'][-1]/xp.data['solpsData']['profiles']['Qtot'][0]))


    # Add some screen output for separatrix and core values
    interp_ne_expt = interpolate.interp1d(xp.data['expData']['fitPsiProf'],xp.data['expData']['fitProfs']['neprof'],kind='linear')
    interp_te_expt = interpolate.interp1d(xp.data['expData']['fitPsiProf'],xp.data['expData']['fitProfs']['teprof'],kind='linear')
    interp_ti_expt = interpolate.interp1d(xp.data['expData']['fitProfs']['tipsi'],xp.data['expData']['fitProfs']['tiprof'],kind='linear')

    interp_ne_solps = interpolate.interp1d(xp.data['solpsData']['psiSOLPS'], xp.data['solpsData']['last10']['ne'])
    interp_te_solps = interpolate.interp1d(xp.data['solpsData']['psiSOLPS'], xp.data['solpsData']['last10']['te'])
    interp_ti_solps = interpolate.interp1d(xp.data['solpsData']['psiSOLPS'], xp.data['solpsData']['last10']['ti'])

    print("\nUsing electron profile shift of: %.3f (in psiN)\n"%elec_prof_rad_shift)
    print("        ne_sep (m^-3)   Te_sep (eV)   Ti_sep (eV)   ne_core (m^-3)")
    print("Expt:    {:.3e}       {:6.2f}        {:6.2f}        {:.3e}".format(interp_ne_expt(1.0-elec_prof_rad_shift)*1e20,
                                                                              interp_te_expt(1.0-elec_prof_rad_shift)*1000,
                                                                              interp_ti_expt(1.0)*1000,
                                                                              interp_ne_expt(np.min(xp.data['solpsData']['psiSOLPS']))*1e20))
    print("SOLPS:   {:.3e}       {:6.2f}        {:6.2f}        {:.3e}\n".format(float(interp_ne_solps(1.0)),
                                                                                float(interp_te_solps(1.0)),
                                                                                float(interp_ti_solps(1.0)),
                                                                                float(interp_ne_solps(np.min(xp.data['solpsData']['psiSOLPS'])))))

    if update_old_last10s:
        update_old_last10_files()        

    return xp

# --- Launch main() ----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    py3_9 = (sys.version_info[0] >= 3 and sys.version_info[1] >= 9)

    parser = argparse.ArgumentParser(description='Generate new b2.transport.inputfile files for SOLPS',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-g', '--gfileloc', help='location of profs_*.pkl saved profile file', type=str, default=None)
    parser.add_argument('-p', '--profilesloc', help='location of profs_*.pkl saved profile file', type=str, default=None)
    parser.add_argument('-s', '--shotnum', help='shot number; default = None', type=str, default=None)
    parser.add_argument('-t', '--timeid', help='time of profile run; default = None', type=str, default=None)
    parser.add_argument('-r', '--runid', help='profile run id; default = None', type=str, default=None)
    parser.add_argument('-i', '--tiratiofile', help='File location for Ti/TD ratio; default = None', type=str, default=None)
    parser.add_argument('-d', '--tdfileloc', help='File location for TD; default = None', type=str, default=None)
    parser.add_argument('-f', '--fractional_change', help='Fractional change to transport coefficients; default = 1',
                        type=float, default=1)
    if py3_9:
        parser.add_argument('--chii_eq_chie', action='store_true', default=False)
        parser.add_argument('--chie_use_grad', action='store_true', default=False)
        parser.add_argument('--chii_use_grad', action='store_true', default=False)
        parser.add_argument('--ti_eq_te', action='store_true', default=False)
        parser.add_argument('--plot_fluxes', action='store_true', default=False)
        parser.add_argument('--update_d_only', action='store_true', default=False)

    args = parser.parse_args()

    if not py3_9:
        args.chii_eq_chie = False
        args.chie_use_grad = False
        args.chii_use_grad = False
        args.ti_eq_te = False
        args.plot_fluxes = False
        args.update_d_only = False

    _ = main(gfile_loc=args.gfileloc, profiles_fileloc=args.profilesloc,
             shotnum=args.shotnum, ptimeid=args.timeid, prunid=args.runid,
             ti_fileloc=args.tdfileloc, ti_eq_te=args.ti_eq_te, update_d_only=args.update_d_only,
             chii_eq_chie=args.chii_eq_chie, chie_use_grad=args.chie_use_grad, chii_use_grad=args.chii_use_grad,
             reduce_Ti_fileloc=args.tiratiofile, fractional_change=args.fractional_change, figblock=True)

# ----------------------------------------------------------------------------------------

def increment_run(new_coefficients = 'b2.transport.inputfile_new', update_old_last10s = True,
                  new_b2xportparams = True, ntim_new = 100, dtim_new = '1.0e-6'):
    """
    After running the main routine, if you're happy with the new transport coefficients,
    run this to save the previous iteration of transport coefficients and prepare to
    run SOLPS with the new coefficients.
    This saves the old b2.transport.inputfile, b2.transport.parameters and b2fstati files
    with the iteration number and updates the b2mn.dat file with short time steps in
    preparation for the new run

    Hide input files from this routine by using '_' in the filename after 'b2.transport.inputfile'

    Inputs:
      new_coefficients    File location for the new b2.transport.inputfile
      update_old_last10s  Set to False to skip saving the .last10 files
      new_b2xportparams   Set to False if you don't want to use the new b2.transport.parameters file
      ntim_new            Number of time steps for b2mn.dat
      dtim_new            Duration of time steps for b2mn.dat
    """
    
    allfiles = os.listdir('.')
    all_incs = [int(i[22:]) for i in allfiles if i[:22] == 'b2.transport.inputfile' and
                i[-1] != '~' and i[-1] != 'e' and i[22] != '_']

    if all_incs:
        inc_num = np.max(all_incs)
    else:
        inc_num = 0
    os.rename('b2.transport.inputfile', 'b2.transport.inputfile' + str(inc_num+1))
    os.system('cp b2fstate b2fstati')
    os.system('cp b2fstate b2fstate' + str(inc_num+1))
    os.rename(new_coefficients, 'b2.transport.inputfile')
    print('Saved previous transport coefficients under iteration ' + str(inc_num+1))
    if new_b2xportparams:
        os.rename('b2.transport.parameters', 'b2.transport.parameters' + str(inc_num+1))
        os.rename('b2.transport.parameters_new', 'b2.transport.parameters')

    # os.remove('run.log')  # Leave this in case there was a mistake or you want to make changes
    if update_old_last10s:
        update_old_last10_files()
    for filename in allfiles:
        if filename[-7:] == '.last10':
            os.remove(filename)
    # os.remove('*.last10')
    # os.system('rm *.last10')  Doesn't work (apparently the star doesn't translate)
    
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

# ----------------------------------------------------------------------------------------

def update_old_last10_files():
    """
    Copy last10 files to .old so that previous profiles can be plotted on next call
    """
    import shutil

    for prof in ['ne', 'dn', 'te', 'ke', 'ti', 'ki']:
        shutil.copyfile(prof+'3da.last10', prof+'3da.last10.old')

# ----------------------------------------------------------------------------------------

def track_inputfile_iterations(rundir=None, impurity_list=['c'], cmap='viridis', Dn_scalar = 100):
    """
    Track the evolution of the b2.transport.inputfile transport
    coefficients through and evolving transport matching job

    Append b2.transport.inputfile filenames with numbers and leave them
    in the same run directory

    Requires everything to be on the same grid with the same species
    """

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

    f, ax = plt.subplots(3, sharex='all')

    for i in range(ninfiles):
        infile = sut.read_b2_transport_inputfile(rundir + inputfile_list[i], carbon=('c' in impurity_list))

        sep_ind = np.argmin(np.abs(infile['rn']))
        dn_sep[i] = infile['dn'][sep_ind]
        ki_sep[i] = infile['ki'][sep_ind]
        ke_sep[i] = infile['ke'][sep_ind]
        dn_bdy[i] = infile['dn'][-1]
        ki_bdy[i] = infile['ki'][-1]
        ke_bdy[i] = infile['ke'][-1]

        ax[0].semilogy(infile['rn'], infile['dn'], '-x', color=cm(i / (float(ninfiles) - 1)),
                       label=inputfile_list[i][13:])
        ax[1].semilogy(infile['rn'], infile['ki'], '-x', color=cm(i / (float(ninfiles) - 1)))
        ax[2].semilogy(infile['rn'], infile['ke'], '-x', color=cm(i / (float(ninfiles) - 1)))

    ax[0].set_ylabel('dn')
    ax[1].set_ylabel('ki')
    ax[2].set_ylabel('ke')
    ax[-1].set_xlabel('rn')
    ax[0].legend(loc='best', fontsize=12)
    for j in range(len(ax)):
        ax[j].grid('on')

    plt.figure()
    plt.semilogy(np.array(range(ninfiles))+1, dn_sep*Dn_scalar, '-xk', lw=2, label='Dn x' + str(Dn_scalar))
    plt.semilogy(np.array(range(ninfiles))+1, ki_sep, '-ob', lw=2, label='ki')
    plt.semilogy(np.array(range(ninfiles))+1, ke_sep, '-or', lw=2, label='ke')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Iteration')
    plt.title('Transport coefficient evolution at separatrix')
    plt.grid('on')

    plt.figure()
    plt.semilogy(np.array(range(ninfiles))+1, dn_bdy*Dn_scalar, '-xk', lw=2, label='Dn x' + str(Dn_scalar))
    plt.semilogy(np.array(range(ninfiles))+1, ki_bdy, '-ob', lw=2, label='ki')
    plt.semilogy(np.array(range(ninfiles))+1, ke_bdy, '-or', lw=2, label='ke')
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('Iteration')
    plt.title('Transport coefficient evolution at boundary')
    plt.grid('on')

    plt.show(block=False)

# ----------------------------------------------------------------------------------------

def plot_matching_case(solps_dir, gfile_loc, profiles_fileloc=None, shotnum=None, ptimeid=None, prunid=None,
                       include_ti=False, plot_psi_mapping = False, b2fgmtry_loc='./b2fgmtry', b2timenc_loc='b2time.nc',
                       xticks=np.arange(0.68, 1.21, 0.04), exp_elec_psin_shift=0, verbose=True):
    """
    Use this to show the final agreement with measurements
    """
    xp = sxp.SOLPSxport(solps_dir, gfile_loc)
    if os.path.isfile('ne3da.last10'):
        xp.getSOLPSlast10Profs()
    else:
        xp.getlast10profs_b2time(b2time_fileloc=b2timenc_loc, reject_fewer_than_10=False)
    xp.loadProfDBPedFit(profiles_file=profiles_fileloc, shotnum=shotnum, timeid=ptimeid, runid=prunid, verbose=verbose)
    xp.populatePedFits()
    b2fgmtry = sut.read_b2fgmtry(b2fgmtry_loc)
    xp.calcPsiVals(plotit=plot_psi_mapping, geo=b2fgmtry, verbose=verbose)

    xp.plot_matching_case(include_ti=include_ti, xticks=xticks, exp_elec_psin_shift=exp_elec_psin_shift)
