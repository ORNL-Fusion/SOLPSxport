"""
This driver is used to generate new b2.transport.inputfile and
b2.transport.parameters files for SOLPS that will come closer to
matching the experimental upstream profiles. If things go well, after
several iterations, this will produce radial ne, Te and Ti profiles in
SOLPS at the outboard midplane that match the experimental profiles
provided. It does this through the generation of an object of class 'SOLPSxport'

The routine "main" runs the sequence in the correct order and returns
the SOLPSxport object for additional plotting and debugging if necessary

**So far this is only coded for cases with D or D+C, no other impurities will work yet**


Instructions for command line call:

-> (if python is not already available) -> $ module load python
-> Source SOLPS-ITER setup file for b2plot calls
-> Navigate to run directory
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -s <shot number> -t <profile time ID> -r <profile run ID>
or if you already have a saved profiles file (.pkl):
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -p <profiles file location>


For calling within interactive Python session:

-> Source SOLPS-ITER setup file for b2plot calls
-> Make sure SOLPSxport_dr is located somewhere in the $PYTHONPATH environment variable
-> Navigate to run directory with b2fstate and other run files
$ ipython
In [1]: import SOLPSxport_dr as sxdr  # (depending on directory structure, might need to be SOLPSxport.SOLPSxport_dr)
In [2]: xp = sxdr.main(gfile_loc = '../g123456.02000', profiles_fileloc = 'prof_123456_2000.pkl')


Requirements:
-Code depends on the SOLPSxport class contained in SOLPSxport.py, as well as routines in SOLPSutils.py
-An existing SOLPS run, which will be used to estimate the next iteration's
radial transport coefficients
-Experimental Te, ne and Ti profiles, preferably saved in the MDSplus database ("Tom's tools")
-Source setup.csh from a SOLPS-ITER distribution that can run b2plot before launching
ipython and importing this module (b2plot is used it grab SOLPS data)

This usually requires several iterations to match the experimental profiles well
(at least ~5 if starting from flat transport coefficients)

Once you've figured out all of the settings you need and are iterating on a run to
converge to the solution for transport coefficients, the routine "increment_run" can
be useful to do this all quickly

R.S. Wilcox, J.M. Canik and J.D. Lore 2020
contact: wilcoxrs@ornl.gov

Reference for this procedure:
https://doi.org/10.1016/j.jnucmat.2010.11.084
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

import SOLPSutils as sut
import SOLPSxport as sxp

plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.facecolor':'w'})
plt.rcParams.update({'mathtext.default': 'regular'})

# ----------------------------------------------------------------------------------------


def main(gfile_loc = None, new_filename='b2.transport.inputfile_new',
         profiles_fileloc=None, shotnum=None, ptimeid=None, prunid=None,
         nefit='tanh', tefit='tanh', ncfit='spl', chii_eq_chie = False, ti_eq_te = False,
         Dn_min=0.001, vrc_mag=0.0, ti_decay_len=0.015, Dn_max=200,
         chie_use_grad = False, chii_use_grad = False, new_b2xportparams = True,
         chie_min = 0.01, chii_min = 0.01, chie_max = 400, chii_max = 400,
         reduce_Ti_fileloc = None,
         fractional_change = 1, exp_prof_rad_shift = 0,
         impurity_list = ['c'], use_existing_last10=False, plot_xport_coeffs=True,
         plotall=False, verbose=False, figblock=False):
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
      chii_eq_chie      Set to True to ignore Ti profile and just set chi_i = chi_e (not implemented yet!!)
      Dn_min            Set a floor for the allowable particle diffusion coefficient
      Dn_max            Set a maximum for the allowable particle diffusion coefficient
      chie_max          Set a maximum for the allowable electron energy diffusion coefficient
      vrc_mag           Hard-coded carbon impurity pinch, for trying to match nC profiles
                        (leave zero unless you also change the function within calcXportCoeffs)
      ti_decay_len      Decay length (at the outboard midplane) for imposed exponential falloff
                        for experimental Ti, beginning at separatrix
                        (since we know Ti measurement from CER is incorrect in SOL)
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
      fractional_change Set to number smaller than 1 if the incremental change is too large and
                        you want to take a smaller step
      exp_prof_rad_shift: Apply a radial shift to experimental profiles
                        (in units of psi_n, positive shifts profiles outward so separatrix is hotter)
      impurity_list     List of all the impurities included in the plasma simulation
                        (not tested yet for anything other than 'c')
      plot_xport_coeffs Plot the SOLPS and experimental profiles, along with the previous
                        and next iteration of transport coefficients
      plotall           Produce a bunch more plots from the subroutines used in the process
      verbose           Set to True to print a lot more outputs to terminal
      figblock          Set to True if calling from command line and you want to see figures

    Returns:
      Object of class 'SOLPSxport', which can then be used to plot, recall, or modify the saved data
      and rewrite a new b2.transport.inputfile
    """

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
    print("Running calcPsiVals")
    xp.calcPsiVals(plotit=plotall)
    print("Running getSOLPSlast10Profs")
    xp.getSOLPSlast10Profs(plotit=plotall, use_existing_last10=use_existing_last10)
    # xp.getProfsOMFIT(prof_folder = prof_folder, prof_filename_prefix = prof_filename_prefix,
    #                  min_npsi = 100, psiMax = 1.05, plotit = plotall)
    if profiles_fileloc is None:  # try to call MDSplus only if no profiles_fileloc is given
        if prunid is not None:
            xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptimeid, prunid, verbose=True)
            print("Populating PedFits from MDSplus database")
            xp.populatePedFits(nemod=nefit, temod=tefit, ncmod=ncfit, npsi=250, plotit=plotall)
        else:
            print("ERROR: Need to provide eiter a profiles fileloc or MDSplus profile runid")
            return
    elif profiles_fileloc[-4:] == '.pkl':
        xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptimeid, prunid, verbose=True)
        print("Populating PedFits")
        xp.populatePedFits(nemod=nefit, temod=tefit, ncmod=ncfit, npsi=250, plotit=plotall)
    # elif profiles_fileloc[:-5] == '.mast':
    #     xp.readMastData(profiles_fileloc)
    else:
        print("Loading profiles from pfile")
        xp.load_pfile(profiles_fileloc, plotit=plotall)
    print("Getting flux profiles")
    xp.getSOLPSfluxProfs(plotit=plotall)

    if impurity_list:
        print("Running getSOLPSCarbonProfs")
        xp.getSOLPSCarbonProfs(plotit=plotall)

    print("Running calcXportCoeff")
    xp.calcXportCoef(plotit=plotall or plot_xport_coeffs, reduce_Ti_fileloc=reduce_Ti_fileloc, Dn_min=Dn_min,
                     ti_decay_len=ti_decay_len, vrc_mag=vrc_mag, verbose=verbose, Dn_max=Dn_max,
                     fractional_change=fractional_change, exp_prof_rad_shift=exp_prof_rad_shift,
                     chii_min=chii_min, chii_max=chii_max, chie_min=chie_min, chie_max=chie_max,
                     chii_eq_chie=chii_eq_chie, figblock=figblock)

    print("Writing to: " + new_filename)
    xp.writeXport(new_filename=new_filename, chie_use_grad=chie_use_grad, chii_use_grad=chii_use_grad,
                  chii_eq_chie=chii_eq_chie)

    # Modify b2.transport.parameters so that PFR has same transport coefficients as first cell of SOL
    if new_b2xportparams:
        psin = list(xp.data['solpsData']['psiSOLPS'])
        first_sol_ind = psin.index(min([i for i in psin if i > 1]))
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
    parser.add_argument('-i', '--tifileloc', help='File location for Ti/TD ratio; default = None', type=str, default=None)
    parser.add_argument('-f', '--fractional_change', help='Fractional change to transport coefficients; default = 1',
                        type=float, default=1)
    if py3_9:
        parser.add_argument('--chii_eq_chie', action='store_true', default=False)
        # parser.set_defaults(chii_eq_chie=False)
        parser.add_argument('--chie_use_grad', action='store_true', default=False)
        # parser.set_defaults(chie_use_grad=False)
        parser.add_argument('--chii_use_grad', action='store_true', default=False)
        # parser.set_defaults(chii_use_grad=True)

    args = parser.parse_args()

    if not py3_9:
        args.chii_eq_chie = False
        args.chie_use_grad = False
        args.chii_use_grad = False

    _ = main(gfile_loc=args.gfileloc, profiles_fileloc=args.profilesloc,
             shotnum=args.shotnum, ptimeid=args.timeid, prunid=args.runid,
             chii_eq_chie=args.chii_eq_chie, chie_use_grad=args.chie_use_grad, chii_use_grad=args.chii_use_grad,
             reduce_Ti_fileloc=args.tifileloc, fractional_change=args.fractional_change, figblock=True)

# ----------------------------------------------------------------------------------------


def increment_run(gfile_loc, new_filename = 'b2.transport.inputfile_new',
                  profiles_fileloc = None, shotnum = None, ptimeid = None, prunid = None,
                  use_existing_last10 = False, chie_use_grad = False, chii_use_grad = False,
                  new_b2xportparams = True,
                  reduce_Ti_fileloc = '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt',
                  impurity_list=['c'], plotall = False, plot_xport_coeffs = True,
                  ntim_new = 100, dtim_new = '1.0e-6', Dn_min = 0.0005):
    """
    This routine runs the main calculation of transport coefficients, then saves the old
    b2.transport.inputfile, b2.transport.parameters and b2fstati files with the iteration
    number and updates the b2mn.dat file with short time steps in preparation for the new run

    Hide input files from this routine by using '_' in the filename after 'b2.transport.inputfile'
    """
    
    xp = main(gfile_loc = gfile_loc, new_filename = new_filename,
              profiles_fileloc = profiles_fileloc, shotnum = shotnum, ptimeid = ptimeid,
              prunid = prunid, Dn_min = Dn_min, use_existing_last10 = use_existing_last10,
              chie_use_grad=chie_use_grad, chii_use_grad=chii_use_grad,
              new_b2xportparams=new_b2xportparams,
              reduce_Ti_fileloc = reduce_Ti_fileloc, impurity_list=impurity_list, plotall = plotall,
              plot_xport_coeffs = plot_xport_coeffs, verbose=False)
    
    allfiles = os.listdir('.')
    all_incs = [int(i[22:]) for i in allfiles if i[:22] == 'b2.transport.inputfile' and
                i[-1] != '~' and i[-1] != 'e' and i[22] != '_']

    if all_incs:
        inc_num = np.max(all_incs)
    else:
        inc_num = 0
    os.rename('b2fstati', 'b2fstati' + str(inc_num+1))
    os.rename('b2.transport.inputfile', 'b2.transport.inputfile' + str(inc_num+1))
    os.rename(new_filename, 'b2.transport.inputfile')
    if new_b2xportparams:
        os.rename('b2.transport.parameters', 'b2.transport.parameters' + str(inc_num+1))
        os.rename('b2.transport.parameters_new', 'b2.transport.parameters')

    # os.remove('run.log')  # Leave this in case there was a mistake or you want to make changes
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

    return xp


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

