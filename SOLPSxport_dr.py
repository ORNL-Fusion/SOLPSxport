"""
This driver is used to generate new b2.transport.inputfile files for SOLPS
that will come closer to matching the experimental upstream profiles.
If things go well, after several iterations, this will produce radial
ne, Te and Ti profiles in SOLPS at the outboard midplane that match the
experimental profiles provided.
It does this through the generation of an object of class 'SOLPSxport'

The routine "main" runs the sequence in the correct order and returns
the SOLPSxport object for additional plotting and debugging if necessary


Instructions for command line call:

->Source SOLPS-ITER setup file for b2plot calls
->Navigate to run directory
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -s <shot number> -t <profile time ID> -r <profile run ID>
or if you already have a saved profiles file (.pkl):
$ python ~/Pytools/SOLPSxport_dr.py -g <gfile location> -p <profiles file location>


Requirements:
-Code depends on the SOLPSxport class contained in SOLPSxport.py, which in turn
depends on routines in SOLPSutils.py
-An existing SOLPS run, which will be used to estimate the next iteration's
radial transport coefficients
-Experimental Te, ne and Ti profiles, preferably saved in the MDSplus database ("Tom's tools")
-Source setup.ksh from a SOLPS-ITER distribution that can run b2plot before launching
ipython and importing this module (b2plot is used it grab SOLPS data)

For best results, use core density boundary condition matched to experimental data
at innermost SOLPS grid location

This usually requires several iterations to match the experimental profiles well

Once you've figured out all of the settings you need and are iterating on a run to
converge to the solution for transport coefficients, the routine "increment_run" can
be useful to do this all quickly

R.S. Wilcox, J.M. Canik and J.D. Lore 2020
contact: wilcoxrs@ornl.gov

Reference for this procedure:
https://doi.org/10.1016/j.jnucmat.2010.11.084
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

import SOLPSxport as sxp

plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.facecolor':'w'})
plt.rcParams.update({'mathtext.default': 'regular'})

# ----------------------------------------------------------------------------------------


def main(gfile_loc = None, new_filename='b2.transport.inputfile_new',
         profiles_fileloc=None, shotnum=None, ptimeid=None, prunid=None,
         nefit='tanh', tefit='tanh', ncfit='spl',
         Dn_min=0.001, vrc_mag=0.0, ti_decay_len=0.015, Dn_max=20,
         ke_use_grad = False, ki_use_grad = True,
         chie_min = 0.01, chii_min = 0.01, chie_max = 200, chii_max = 200,
         reduce_Ti_fileloc='/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt',
         carbon=True, use_existing_last10=False, plot_xport_coeffs=True,
         plotall=False, verbose=False, figblock=False):
    """
    Driver for the code using Osborne profile fits saved in MDSplus

    Inputs:
      rundir            Location of SOLPS run directory
                        --> depricated, assume you are in the directory (this is required for b2plot calls)
      gfile_loc         Location of gfile used for SOLPS grid (for mapping SOLPS grid to psin)
      profiles_fileloc  (optional) Location of a .pkl file with saved Tom's tools profile
                        fit coefficients (produced by getProfDBPedFit from SOLPSutils)
      shotnum,ptimeid,prunid  Profile identifying shot number, timeid and runid (from Tom's tools)
                        (this is uneccessary if you have a .pkl file given in profiles_fileloc)
      xxfit             Fit function used in each profile fit (xx => ne, te and nc)
      Dn_min            Set a floor for the allowable particle diffusion coefficient
      Dn_max            Set a maximum for the allowable particle diffusion coefficient
      chie_max          Set a maximum for the allowable electron energy diffusion coefficient
      vrc_mag           Hard-coded carbon impurity pinch, for trying to match nC profiles
                        (leave zero unless you also change the function within calcXportCoeffs)
      ti_decay_len      Decay length (at the outboard midplane) for imposed exponential falloff
                        for experimental Ti, beginning at separatrix
                        (since we know Ti measurement from CER is incorrect in SOL)
      ke/i_use_grad     Use ratio of the gradients for new values of chi_e/i, rather than fluxes
                        For some reason I don't understand (bug?), flux formula doesn't work well for chi_i
      use_existing_last10  Set to True if you have already run 2d_profiles to produce .last10 files
                           in the run folder to save time. Otherwise this will call 2d_profiles so
                           that you don't accidentally use .last10 files from a previous iteration
                           with out-of-date SOLPS profiles
      reduce_Ti_fileloc Set to None to use T_D = T_C from MDS+ profile fit
      carbon            Set to False if SOLPS run includes D only
                        note: this routine is not yet generalized to anything other than D or D+C
      plot_xport_coeffs Plot the SOLPS and experimental profiles, along with the previous
                        and next iteration of transport coefficients
      plotall           Produce a bunch more plots from the subroutines used in the process
      verbose           Set to True to print a lot more outputs to terminal
      figblock          Set to True if calling from command time and you want to see figures

    Returns:
      Object of class 'SOLPSxport', which can then be called to plot or recall the saved data
    """

    if shotnum is None: shotnum = int(gfile_loc[-12:-6])
    if ptimeid is None: ptimeid = int(gfile_loc[-4:])
    ptimeid = int(ptimeid)
    shotnum = int(shotnum)

    print("Initializing SOLPSxport")
    xp = sxp.SOLPSxport(workdir=os.getcwd(), gfile_loc=gfile_loc, carbon_bool=carbon)
    print("Running calcPsiVals")
    xp.calcPsiVals(plotit=plotall)
    print("Running getSOLPSlast10Profs")
    xp.getSOLPSlast10Profs(plotit=plotall, use_existing_last10=use_existing_last10)
    xp.loadProfDBPedFit(profiles_fileloc, shotnum, ptimeid, prunid, verbose=True)
    print("Populating PedFits")
    xp.populatePedFits(nemod=nefit, temod=tefit, ncmod=ncfit, npsi=250, plotit=plotall)
    print("Getting flux profiles")
    xp.getSOLPSfluxProfs(plotit=plotall)

    if carbon:
        print("Running getSOLPSCarbonProfs")
        xp.getSOLPSCarbonProfs(plotit=plotall)

    print("Running calcXportCoeff")
    xp.calcXportCoef(plotit=plotall or plot_xport_coeffs, reduce_Ti_fileloc=reduce_Ti_fileloc, Dn_min=Dn_min,
                     ti_decay_len=ti_decay_len, vrc_mag=vrc_mag, verbose=verbose, Dn_max=Dn_max,
                     chii_min=chii_min, chii_max=chii_max, chie_min=chie_min, chie_max=chie_max, figblock=figblock)

    print("Running writeXport")
    xp.writeXport(new_filename=new_filename, ke_use_grad=ke_use_grad, ki_use_grad=ki_use_grad)

    return xp

# --- Launch main() ----------------------------------------------------------------------


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate new b2.transport.inputfile files for SOLPS',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-g', '--gfileloc', help='location of profs_*.pkl saved profile file', type=str, default=None)
    parser.add_argument('-p', '--profilesloc', help='location of profs_*.pkl saved profile file', type=str, default=None)
    parser.add_argument('-s', '--shotnum', help='shot number; default = None', type=str, default=None)
    parser.add_argument('-t', '--timeid', help='time of profile run; default = None', type=str, default=None)
    parser.add_argument('-r', '--runid', help='profile run id; default = None', type=str, default=None)
    parser.add_argument('-i', '--tifileloc', help='File location for Ti/TD ratio; default = None', type=str, default=None)

    args = parser.parse_args()

    _ = main(gfile_loc=args.gfileloc, profiles_fileloc=args.profilesloc,
             shotnum=args.shotnum, ptimeid=args.timeid, prunid=args.runid,
             reduce_Ti_fileloc=args.tifileloc, figblock=True)

# ----------------------------------------------------------------------------------------


def main_omfit(topdir, subfolder, gfile_loc, prof_folder = None,
               prof_filename_prefix = 'prof171558_3200',
               new_filename = 'b2.transport.inputfile_new',
               use_existing_last10 = False,
               carbon = True, plotall = False, debug_plots = False, plot_xport_coeffs = True):
    """
    **This has not yet been fixed to work with the current version of these codes**
    """
    print("WARNING: This routine is likely to break")
    print("         Updates need to be made before it works with OMFIT outputs")

    print("Initializing SOLPSxport")
    xp = sxp.SOLPSxport(workdir = topdir + subfolder, gfile_loc = gfile_loc, carbon_bool = carbon)
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

    print("Writing to " + new_filename)
    xp.writeXport(new_filename = new_filename)

    return xp

# ----------------------------------------------------------------------------------------


def increment_run(gfile_loc, new_filename = 'b2.transport.inputfile_new',
                  profiles_fileloc = None, shotnum = None, ptimeid = None, prunid = None,
                  use_existing_last10 = False, ke_use_grad = False, ki_use_grad = True,
                  reduce_Ti_fileloc = '/fusion/projects/results/solps-iter-results/wilcoxr/T_D_C_ratio.txt',
                  carbon = True, plotall = False, plot_xport_coeffs = True,
                  ntim_new = 100, dtim_new = '1.0e-6', Dn_min = 0.0005):
    """
    This routine runs the main calculation of transport coefficients, then saves
    the old b2.transport.inputfile and b2fstati files with the iteration number
    and updates the b2mn.dat file with short time steps in preparation for the new run

    Hide input files from this routine by using '_' in the filename after 'b2.transport.inputfile'
    """
    
    xp = main(gfile_loc = gfile_loc, new_filename = new_filename,
              profiles_fileloc = profiles_fileloc, shotnum = shotnum, ptimeid = ptimeid,
              prunid = prunid, Dn_min = Dn_min, use_existing_last10 = use_existing_last10,
              ke_use_grad=ke_use_grad, ki_use_grad=ki_use_grad,
              reduce_Ti_fileloc = reduce_Ti_fileloc, carbon = carbon, plotall = plotall,
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


def track_inputfile_iterations(rundir=None, carbon=True, cmap='viridis', Dn_scalar = 100):
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

    f, ax = plt.subplots(3, sharex='all')

    for i in range(ninfiles):
        infile = read_b2_transport_inputfile(rundir + inputfile_list[i], carbon=carbon)

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
