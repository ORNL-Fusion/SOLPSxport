"""
collection of utility routines for use with other SOLPS modules

Several routines that used to be here have simpler/more robust duplicates in common libraries
I've left those routines below but commented them out and given a good replacement option

A. Sontag, R.S. Wilcox 2019
"""

from os import path, system, rename
import numpy as np


# ----------------------------------------------------------------------------------------

def calcTanhMulti(c, x, param=None):
    """
    tanh function with cubic or quartic inner and linear to quadratic outer extensions
    and derivative = 0 at param
    
    0.5*(c[2]-c[3])*(pz1*exp(z)-pz2*exp(-z))/(exp(z)+exp(-z))+0.5*(c[2]+c[3])
    where z=2*(c[0]-x)/c[1]
    
    if param=None:
        pz1=1+c[4]*z+c[5]*z*z+c[6]*z*z*z
    else:
        pz1=1+cder*z+c[4]*z*z+c[5]*z*z*z+c[6]*z*z*z*z
        where cder=-(2.0*c[4]*z0+3.0*c[5]*z0*z0+4.0*c[6]*z0*z0*z0
        and z0=2.0*(c[0]-param)/c[1]
        
        pz2=1+(c[7]*z+c[8]*z*z) depending on whether there are 7, 8 or 9 coefs
    
    c0 = SYMMETRY POINT
    c1 = FULL WIDTH
    c2 = HEIGHT
    c3 = OFFSET
    c4 = SLOPE OR QUADRATIC (IF ZERO DERIV) INNER
    c5 = QUADRATIC OR CUBIC (IF ZERO DERIV) INNER
    c6 = CUBIC OR QUARTIC (IF ZERO DERIV) INNER
    c7 = SLOPE OF OUTER
    c8 = QUADRATIC OUTER
    
    ** translated from IDL by A. Sontag 4-4-18
    """

    z = 2 * (c[0] - x) / c[1]
    out = np.zeros(len(z))

    if len(c) == 5:
        for i in range(0, len(z)):
            out[i] = 0.5 * (c[2] - c[3]) * ((1 + c[4] * z[i]) * np.exp(z[i]) - np.exp(-z[i])) / \
                     (np.exp(z[i]) + np.exp(-z[i])) + 0.5 * (c[2] + c[3])
    elif len(c) == 6:
        # pz1 = np.zeros(len(z))
        if param:
            z0 = 2 * (c[0] - param) / c[1]
            cder = -(2 * c[3] * z0 + 3 * c[4] * z0**2 + 4 * c[5] * z0**3)
            pz1 = 1 + cder * z + c[3] * z**2 + c[4] * z**3 + c[5] * z**4
        else:
            pz1 = 1 + c[3] * z + c[4] * z**2 + c[5] * z**3
        for i in range(0, len(z)):
            out[i] = 0.5*c[2]*(pz1[i]*np.exp(z[i]) - np.exp(-z[i])) / \
                              (np.exp(z[i]) + np.exp(-z[i])) + 0.5*c[2]
    else:
        # pz1 = np.zeros(len(z))
        if param:
            z0 = 2 * (c[0] - param) / c[1]
            cder = -(2 * c[4] * z0 + 3 * c[5] * z0**2 + 4 * c[6] * z0**3)
            pz1 = 1 + cder * z + c[4] * z**2 + c[5] * z**3 + c[6] * z**4
        else:
            pz1 = 1 + c[4] * z + c[5] * z**2 + c[6] * z**3

        pz2 = np.ones(len(z))
        if len(c) > 7: pz2 += c[7] * z
        if len(c) > 8: pz2 += c[8] * z**2

        for i in range(0, len(z)):
            out[i] = 0.5 * (c[2] - c[3]) * (pz1[i] * np.exp(z[i]) - pz2[i] * np.exp(-z[i])) / \
                           (np.exp(z[i]) + np.exp(-z[i])) + 0.5 * (c[2] + c[3])

    return out

# ----------------------------------------------------------------------------------------            


def loadMDS(tree, tag, shot, quiet=True):
    import MDSplus

    c = MDSplus.Connection('atlas.gat.com')
    c.openTree(tree, shot)

    try:
        y = c.get(tag).data()
    except:
        print('invalid data for ' + tag)
        y = None

    try:
        x = c.get('DIM_OF(' + tag + ')').data()
    except:
        x = None

    try:
        yerr = c.get('ERROR_OF(' + tag + ')').data()
    except:
        yerr = None

    try:
        xerr = c.get('ERROR_OF(DIM_OF(' + tag + '))').data()
    except:
        xerr = None

    out = dict(x=x, y=y, xerr=xerr, yerr=yerr)

    if not quiet: print('done with ' + tag)

    return out

# ----------------------------------------------------------------------------------------


def B2pl(cmds, wdir='.', debug=False):
    """
    runs B2plot with the commands used in the call and reads contents of the resulting
    b2plot.write file into two lists
    
    ** Make sure you've sourced the setup script first, or this won't work! **
    **  Make sure B2PLOT_DEV is set to 'ps'
    """

    if debug:
        cmdstr = 'echo "' + cmds + '" | b2plot'
        print(cmdstr)
    else:
        cmdstr = 'echo "' + cmds + '" | b2plot >&/dev/null'
    system(cmdstr)

    fname = path.join(wdir, 'b2pl.exe.dir', 'b2plot.write')
    if not path.exists(fname):
        print('B2Plot writing failed for call:')
        print(cmds)
        print('in directory: ' + wdir + '\n')

    x, y = [], []
    with open(fname) as f:
        lines = f.readlines()

    for line in lines:
        elements = line.split()
        if elements[0] == '#':
            pass
        else:
            x.append(float(elements[0]))
            y.append(float(elements[1]))
    x = x[0:(len(x) // 2)]  # used to be: x=x[0:(len(x)/2)-1], chopped final value
    y = y[0:(len(y) // 2)]

    return x, y

# ----------------------------------------------------------------------------------------


def readProf(fname, wdir='.'):
    """
    reads contents of text file into two lists, returns them as numpy arrays
    """

    fname = path.join(wdir, fname)
    x, y = [], []

    with open(fname) as f:
        lines = f.readlines()

    for line in lines:
        elements = line.split()

        if elements[0] == '#':
            pass
        else:
            x.append(float(elements[0]))
            y.append(float(elements[1]))

    return x, y

# ----------------------------------------------------------------------------------------


def loadg(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()

    # read first line for case string and grid size
    line = lines[0]
    words = line.split()

    nw = int(words[-2])
    nh = int(words[-1])
    psi = np.linspace(0, 1, nw)

    # read in scalar parameters
    #   note: word size of 16 characters each is assumed for all of the data to be read

    # line 1
    line = lines[1]
    rdim = float(line[0:16])
    zdim = float(line[16:32])
    rcentr = float(line[32:48])
    rleft = float(line[48:64])
    zmid = float(line[64:80])

    # line 2
    line = lines[2]
    rmaxis = float(line[0:16])
    zmaxis = float(line[16:32])
    simag = float(line[32:48])
    sibry = float(line[48:64])
    bcentr = float(line[64:80])

    # line 3
    line = lines[3]
    current = float(line[0:16])

    # read in profiles
    #   start by reading entire file into single list then split into individual profiles
    #   first block has 5 profiles of length nw and one array of length nh*nw

    temp = []
    count = 0
    lnum = 5
    terms = 5 * nw + nw * nh
    while count < terms:
        line = lines[lnum]
        numchar = len(line)
        nwords = numchar // 16
        count1 = 0
        while count1 < nwords:
            i1 = count1 * 16
            i2 = i1 + 16
            temp.append(float(line[i1:i2]))
            count1 += 1
            count += 1
        lnum += 1

    fpol = temp[0:nw]
    pres = temp[nw:2 * nw]
    ffprime = temp[2 * nw:3 * nw]
    pprime = temp[3 * nw:4 * nw]
    psirz_temp = temp[4 * nw:(4 + nh) * nw]
    qpsi = temp[(4 + nh) * nw:]

    # split psirz up into 2D matrix
    count = 0
    psirz = []
    while count < nh:
        ind1 = count * nw
        ind2 = ind1 + nw
        psirz.append(psirz_temp[ind1:ind2])
        count += 1

    # scalars for length of boundary and limiter arrays
    line = lines[lnum]
    words = line.split()
    nbbbs = int(words[0])
    limitr = int(words[1])

    # read boundary and limiter points into temp array

    temp = []
    count = 0
    terms = 2 * (nbbbs + limitr)
    lnum += 1
    while count < terms:
        line = lines[lnum]
        numchar = len(line)
        nwords = numchar // 16
        count1 = 0
        while count1 < nwords:
            i1 = count1 * 16
            i2 = i1 + 16
            temp.append(float(line[i1:i2]))
            count1 += 1
            count += 1
        lnum += 1
    bdry_temp = temp[0:(2 * nbbbs)]
    limit_temp = temp[(2 * nbbbs):]

    # split boundary into (R,Z) pairs
    count = 0
    rbdry = []
    zbdry = []
    while count < len(bdry_temp) - 1:
        rbdry.append(bdry_temp[count])
        zbdry.append(bdry_temp[count + 1])
        count += 2

    # split limiter into (R,Z) pairs
    count = 0
    rlim = []
    zlim = []
    while count < len(limit_temp) - 1:
        rlim.append(limit_temp[count])
        zlim.append(limit_temp[count + 1])
        count += 2

    g = dict(nw=nw, nh=nh, rdim=rdim, zdim=zdim, rcentr=rcentr, rleft=rleft, zmid=zmid,
             rmaxis=rmaxis, zmaxis=zmaxis, simag=simag, sibry=sibry, current=current,
             fpol=np.array(fpol),
             ffprime=np.array(ffprime), pprime=np.array(pprime), psirz=np.array(psirz),
             qpsi=np.array(qpsi), nbbbs=nbbbs, bcentr=bcentr,
             pres=np.array(pres), limitr=limitr, rbdry=np.array(rbdry),
             zbdry=np.array(zbdry), rlim=np.array(rlim), zlim=np.array(zlim))

    return g


# ----------------------------------------------------------------------------------------

def list2H5(data, pathname, outname):
    import h5py

    outname += '.h5'
    out = h5py.File(path.join(pathname, outname), 'w')

    var = data.keys()

    for v in var:
        vals = data[v]
        try:
            dset = out.create_dataset(v, np.shape(vals), 'f', compression='gzip', shuffle='true')
            dset[...] = vals
        except:
            pass

    out.close()


# ----------------------------------------------------------------------------------------


def getProfDBPedFit(shotnum, timeid, runid, write_to_file=None):
    """
    Loads saved data from Tom's tools MDSplus server
     'XXdatpsi' :  Raw data
     
     write_to_file: Give file name (prefer extension '.txt')
                    .pkl files using pickle module are the old format, but these can break if
                    you ever use a different version of Python to read than what was used to write.
    """

    tree = 'profdb_ped'

    tagList = ['nedatpsi', 'tedatpsi', 'tidatpsi', 'netanhpsi', 'ttst',
               'netanhpsi:fit_coef', 'tetanhpsi:fit_coef', 'titanhpsi:fit_coef',
               'tisplpsi', 'ptotsplpsi', 'zfz1datpsi', 'zfz1splpsi']

    # tagList=['nedatpsi','tedatpsi','tidatpsi','netanhpsi','fzdatpsi','zfz1datpsi',
    # 'vtordatpsi','ttst','netnh0psi:fit_coef','netanhpsi:fit_coef','tetnh0psi:fit_coef',
    # 'tetanhpsi:fit_coef','titanhpsi:fit_coef','tisplpsi','ptotsplpsi','vtorsplpsi',
    # 'fzsplpsi','zfz1splpsi']

    profile_fits = {}
    tstr = ':p' + str(timeid) + '_' + runid + ':'
    for t in tagList:
        tag = tstr + t
        val = loadMDS(tree, tag, shotnum)
        if t[-9:] == ':fit_coef': t = t[:-9]
        profile_fits[t] = val

    if write_to_file is not None:
        if write_to_file[-4:] == '.pkl':
            import pickle

            with open(write_to_file, 'wb') as f:
                pickle.dump(profile_fits, f, pickle.HIGHEST_PROTOCOL)
        else:
            print("can't print to that file type yet, only .pkl")

    return profile_fits


# ----------------------------------------------------------------------------------------


def read_pfile(pfile_loc):
    """
    Read in the kinetic profiles from a p file to be used as inputs (successfully tested 2018/1/3)

    Returns a dictionary with a non-intuitive set of keys (units are included)
    
    ** Note: pfiles don't normally go into the SOL **
    """
    with open(pfile_loc, mode='r') as pfile:
        lines = pfile.readlines()

    profiles = {}
    nprofs = 0  # counter for total number of profiles so far
    linestart = 0  # counter for which line to start at for each profile
    nlines_tot = len(lines)

    while True:
        # Read the header line for each profile first
        lin1 = lines[linestart].split()
        npts_prof = int(lin1[0])

        xname = lin1[1]
        yname = lin1[2]
        dyname = ''.join(lin1[3:])[:-1]

        # Generate and populate the profile arrays
        x = np.zeros(npts_prof)
        y = np.zeros(npts_prof)
        dy = np.zeros(npts_prof)
        for i in range(npts_prof):
            split_line = lines[linestart + i + 1].split()
            x[i] = float(split_line[0])
            y[i] = float(split_line[1])
            dy[i] = float(split_line[2][:-1])

        # profiles[xname + '_' + yname] = x  # psinorm
        profiles[xname] = x
        profiles[yname] = y
        profiles[dyname] = dy

        nprofs += 1
        linestart += 1 + npts_prof

        if linestart >= nlines_tot:
            break

    # Check if all psinorms are the same, consolidate if so (they are, don't bother separating)

    # condense = True
    # psinorm = None
    # for k in profiles.keys():
    #     if k is None or k=='':
    #         continue
    #
    #     if k[:4] == 'psin':
    #         if psinorm is None:
    #             psinorm = profiles[k]
    #
    #         if max(abs(profiles[k] - psinorm)) > 1e-5:
    #             condense = False
    #             break

    # if condense:
    #     profiles = {key: value for key, value in profiles.items()
    #                 if key[:4] != 'psin' or key is None or key==''}
    #     profiles['psinorm'] = psinorm

    return profiles

# ----------------------------------------------------------------------------------------


def read_b2_transport_inputfile(infileloc, carbon=True):
    """
    Reads b2.transport.inputfile, outputs basic quantities as a dictionary
    
    All carbon species are assumed to have the same transport coefficients
    (this could be fixed easily if you want)
    
    !!!WARNING!!!
      This was only written to read inputfiles written using SOLPSxport.py,
      and therefore may not work properly if your file is formatted differently.
    """
    with open(infileloc, 'r') as f:
        lines = f.readlines()

    ndata = int(lines[1].strip().split()[5])  # This is the same for every array in our write routine

    rn = np.zeros(ndata)
    dn = np.zeros(ndata)
    ke = np.zeros(ndata)
    ki = np.zeros(ndata)
    if carbon:
        vrc = np.zeros(ndata)
        dc = np.zeros(ndata)

    for line_full in lines[2:]:
        line = line_full.strip().split()
        if len(line) < 4: continue

        if line[2][0] == '1' and line[3][0] == '1':
            rn[int(line[1][:-1]) - 1] = np.float(line[5])
            dn[int(line[1][:-1]) - 1] = np.float(line[-2])

        elif line[2][0] == '3' and line[3][0] == '1':
            ki[int(line[1][:-1]) - 1] = np.float(line[-2])

        elif line[2][0] == '4' and line[3][0] == '1':
            ke[int(line[1][:-1]) - 1] = np.float(line[-2])

        elif carbon:

            if line[2][0] == '1' and line[3][0] == '4':
                dc[int(line[1][:-1]) - 1] = np.float(line[-2])

            elif line[2][0] == '6' and line[3][0] == '3':
                vrc[int(line[1][:-1]) - 1] = np.float(line[-2])

    if carbon:
        return {'rn': rn, 'dn': dn, 'ki': ki, 'ke': ke, 'dc': dc, 'vrc': vrc}
    else:
        return {'rn': rn, 'dn': dn, 'ki': ki, 'ke': ke}

# ----------------------------------------------------------------------------------------


def read_b2fgmtry(fileloc):
    """
    Modified from omfit_solps.py
    """

    with open(fileloc, 'r') as f:
        tmp = f.read()
    tmp = tmp.replace('\n', ' ')
    tmp = tmp.split("*c")
    tmp = [[f for f in x.split(' ') if f] for x in tmp]

    m = {'int': int, 'real': float, 'char': str}

    b2fgmtry = {}

    for line in tmp[1:]:
        if len(line) > 4:
            b2fgmtry[line[3]] = np.array(list(map(m[line[1]], line[4:])))
        else:
            b2fgmtry[line[3]] = None

    return b2fgmtry


# ----------------------------------------------------------------------------------------

def new_b2xportparams(fileloc, dperp=None, chieperp=None, chiiperp=None, verbose=False, ndigits=8):
    """
    Update b2.transport.parameters file with new transport coefficients
    Leaves old file in place, produces new file with appended name '_new'

    Inputs:
      fileloc    Should end in 'b2.transport.parameters' unless you're doing something weird
      dperp      Perpendicular particle diffusion coefficient (will not be modified if left as None)
      chieperp   Perpendicular electron thermal diffusion coefficient (will not be modified if left as None)
      chiiperp   Perpendicular ion thermal diffusion coefficient (will not be modified if left as None)
      ndigits    Number of digits beyond the decimal point to include before rounding

    Expected format, from an example file from DIII-D:

     &transport
     write_nml_transp = .false.,
     flag_dna=1, parm_dna=9*0.03,
     flag_dpa=1, parm_dpa=9*0.0,
     flag_vla=1, parm_vla=9*0.0,
     flag_vsa=1, parm_vsa=9*0.2,
     flag_hci=1, parm_hci=9*5.0,   # Ti is the same for all species, but n changes per species, so chi can be different
     flag_hce=1, parm_hce=5.0,
     flag_sig=1, parm_sig=0.000001,
     flag_alf=1, parm_alf=0.000001,
     /
    """
    if fileloc[-23:] != 'b2.transport.parameters':
        print("WARNING: trying to modify something that should be named 'b2.transport.parameters', but it has a different name")

    with open(fileloc, 'r') as f:
        lines = f.readlines()

    for i, l in enumerate(lines):

        if dperp is not None:
            if 'parm_dna' in l:
                parm_ind = l.rfind('parm_dna')
                if '*' in l[parm_ind:]:
                    mult_ind = l.rfind('*')
                    lines[i] = l[:mult_ind+1] + str(round(dperp, ndigits)) + ',\n'
                    continue
                else:
                    print('WARNING: Unexpected file format for b2.transport.parameters')
                    print('Not modifying b2.transport.parameters, so check PFR')
                    return

        if chiiperp is not None:   # test this, should be ok
            if 'parm_hci' in l:
                parm_ind = l.rfind('parm_hci')
                if '*' in l[parm_ind:]:
                    mult_ind = l.rfind('*')
                    lines[i] = l[:mult_ind + 1] + str(round(chiiperp, ndigits)) + ',\n'
                    continue
                else:
                    print('WARNING: Unexpected file format for b2.transport.parameters')
                    print('Not modifying b2.transport.parameters, so check PFR')
                    return

        if chieperp is not None:
            if 'parm_hce' in l:
                eq_ind = l.rfind('=')
                lines[i] = l[:eq_ind + 1] + str(round(chieperp, ndigits)) + ',\n'

    # rename(fileloc, fileloc + '_old')
    if verbose:
        if fileloc[:2] == './':
            fileloc_print = fileloc[2:]
        else:
            fileloc_print = fileloc
        print('New version of ' + fileloc_print + ' generated: ' + fileloc_print + '_new')

    with open(fileloc + '_new', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])


# ----------------------------------------------------------------------------------------

def read_b2fstat(fileloc):
    """
    Modified from omfit_solps.py
    """
    with open(fileloc, 'r') as f:
        lines = f.readlines()

    b2fstat = {'__notes__': [], '__unhandled__': []}

    # Initialize
    data = np.array([])

    translations = {'real': 'float', 'char': 'str'}

    b2fstat['nx'], b2fstat['ny'], b2fstat['ns'] = 0, 0, 0

    def close_field():
        # Handle previous field
        if n > 0:
            print('Finishing previous field;    dat_type = "{}", n = {}, ' +
                  'names = {}, len(names) = {}'.format(dat_type, n, names, len(names)))
            if len(names) == 1:
                # 2D arrays: X Y
                if n == (b2fstat['nx'] + 2) * (b2fstat['ny'] + 2):
                    # This is an X by Y spatial array with guard cells
                    b2fstat[names[0]] = data.reshape(-1, b2fstat['nx'] + 2)
                elif n == b2fstat['nx'] * b2fstat['ny']:
                    # This is an X by Y spatial array with no guard cells
                    b2fstat[names[0]] = data.reshape(-1, b2fstat['nx'])

                # 3D arrays - X Y S
                elif n == (b2fstat['nx'] + 2) * (b2fstat['ny'] + 2) * b2fstat['ns']:
                    # This is an X by Y by species array with guard cells
                    b2fstat[names[0]] = data.reshape(b2fstat['ns'], -1, b2fstat['nx'] + 2)
                elif n == (b2fstat['nx']) * (b2fstat['ny']) * b2fstat['ns']:
                    # This is an X by Y by species array with no guard cells
                    b2fstat[names[0]] = data.reshape(b2fstat['ns'], -1, b2fstat['nx'])

                # 3D arrays - X Y v
                # Page 183 of the 2015 Feb 27 SOLPS manual mentions "fhe - (-1:nx, -1:ny, 0:1) real*8 array
                # I think there's a row major vs. column major difference going on or something, but anyway,
                # python gets the array out correctly if nx corresponds to the last axis when 2D, which is
                # the opposite of the documentation, so the v axis with length 2 should be the first axis in
                # python.
                elif n == (b2fstat['nx'] + 2) * (b2fstat['ny'] + 2) * 2:
                    # This is an X by Y by 2 (probably poloidal & radial components) array w/ guard cells.
                    b2fstat[names[0]] = data.reshape(2, -1, b2fstat['nx'] + 2)
                elif n == b2fstat['nx'] * b2fstat['ny'] * 2:
                    # This is an X by Y by 2 (probably poloidal & radial components) array w/o guard cells.
                    b2fstat[names[0]] = data.reshape(2, -1, b2fstat['nx'])

                # 4D arrays - X Y S v
                # Manual calls fna X Y v S, so it should be S v Y X here.
                elif n == (b2fstat['nx'] + 2) * (b2fstat['ny'] + 2) * b2fstat['ns'] * 2:
                    # This is an X by Y by species by 2 array w/ guard cells.
                    b2fstat[names[0]] = data.reshape(b2fstat['ns'], 2, -1, b2fstat['nx'] + 2)
                elif n == (b2fstat['nx']) * (b2fstat['ny']) * b2fstat['ns'] * 2:
                    # This is an X by Y by species by 2 array w/o guard cells.
                    b2fstat[names[0]] = data.reshape(b2fstat['ns'], 2, -1, b2fstat['nx'])

                else:
                    # Can't identify higher dims of this or it is just 1D, so don't mess with it
                    b2fstat[names[0]] = data
            elif len(names) == n:
                for ii, name in enumerate(names):
                    b2fstat[name] = data[ii]
            else:
                print('WARNING! Problem parsing b2fstate or b2fstati in omfit_solps class: ' +
                      'Cannot handle more than one name unless length of names matches length' +
                      ' of data!')
                b2fstat['__unhandled__'] += [{'names': names, 'data': data}]

    cflines = []
    for i, line in enumerate(lines):
        if line.startswith('*cf:'):
            cflines += [i]

    cflines += [len(lines)]

    if cflines[0] > 0:
        print('    Found header line(s), adding to notes: {}'.format(lines[: cflines[0]]))
        b2fstat['__notes__'] += [line.split('\n')[0] for line in lines[: cflines[0]]]

    for i in range(len(cflines) - 1):
        print('Setting up field: {}'.format(lines[cflines[i]].split('\n')[0]))
        dat_type_string = lines[cflines[i]].split()[1].strip()
        dat_type_string = translations.get(dat_type_string, dat_type_string)
        dat_type = eval(dat_type_string)
        n = int(lines[cflines[i]].split()[2])
        names = lines[cflines[i]].split()[3].split(',')
        print('   dat_type = "{}" / "{}", n = {}, names = {}, len(names) = {}'.format(
              dat_type_string, dat_type, n, names, len(names)),)

        raw = lines[cflines[i] + 1: cflines[i + 1]]
        if dat_type_string != 'char':
            data = np.array(' '.join(raw).split()).astype(dat_type)
        else:
            data = lines[cflines[i] + 1].split('\n')[0][1:]
        close_field()

    return b2fstat

# ----------------------------------------------------------------------------------------


def read_input_dat(fileloc, verbose = False):
    """
    Read and parse the input.dat file

    For now just reads the EIRENE surface locations and albedo, but can be easily appended to include more
    """
    if not (fileloc[-9:] == 'input.dat' or fileloc[-9:] == 'input.eir'):
        print('WARNING: expected file name "input.dat"')

    if verbose:
        print("Reading input.dat file from: " + fileloc)

    with open(fileloc, 'r') as f:
        lines = f.readlines()

    b3bind = 0
    surfmod1_ind = 0
    for i, l in enumerate(lines):
        if l[:6] == "*** 3b":
            b3bind = i  # starting index of block 3b
        if l[:6] == "*** 6a":
            for j, surfmodblocklines in enumerate(lines[i:]):
                if surfmodblocklines[:9] == 'SURFMOD_1':
                    surfmod1_ind = i + j  # starting index of SURFMOD_1 line
                    break
            break

    if b3bind == 0:
        print("No block labeled '3b' in text file provided, check that it's a SOLPS input.dat file: " + fileloc)
        return

    nsurfs = int(lines[b3bind+1])

    rlocs1 = np.zeros(nsurfs)
    zlocs1 = np.zeros(nsurfs)
    rlocs2 = np.zeros(nsurfs)
    zlocs2 = np.zeros(nsurfs)
    surfmod = np.zeros(nsurfs)
    for i in range(nsurfs):
        rlocs1[i] = float(lines[b3bind + 5*(i+1)][:12]) / 100.0  # in m
        zlocs1[i] = float(lines[b3bind + 5*(i+1)][12:24]) / 100.0
        rlocs2[i] = float(lines[b3bind + 5*(i+1)][36:48]) / 100.0
        zlocs2[i] = float(lines[b3bind + 5*(i+1)][48:60]) / 100.0
        surfmod[i] = int(lines[b3bind + 5*(i+1) + 1][8])

    nsurfmods = int(np.max(surfmod))

    recyct = np.zeros(nsurfmods)
    for i in range(nsurfmods):
        recyct[i] = float(lines[surfmod1_ind + 5*i + 3][13:24])

    return {'nsurfs':nsurfs, 'rlocs1':rlocs1, 'zlocs1':zlocs1, 'rlocs2':rlocs2, 'zlocs2':zlocs2,
            'surfmod':surfmod, 'recyct':recyct}

# ----------------------------------------------------------------------------------------
# def shift(arr,ns):
#     """
#     Use np.roll for this instead ('append' doesn't work for numpy arrays)
#     2nd argument needs to be opposite for np.roll (2 --> -2)
#     """
#     print("Use np.roll instead of this custom 'shift' routine")
#     new_arr=arr[ns:]
#     for e in arr[:ns]:
#         new_arr.append(e)
#
#     return new_arr
#
# ----------------------------------------------------------------------------------------
# def deriv(x,y):
#     """
#     This function duplicates the deriv finite differencing function from IDL
#
#     Use np.gradient(y) / np.gradient(x) instead of this, it handles end indeces properly
#     """
#
#     xx = np.array(x)
#     y = np.array(y)
#
#     x12 = xx - np.roll(xx,1)
#     x01 = np.roll(xx,-1) - xx
#     x02 = np.roll(xx,-1) - np.roll(xx,1)
#
#     d = np.roll(y,-1)*(x12/(x01*x02))+y*(1/x12-1/x01)-np.roll(y,1)*(x01/(x02*x12))
#
#     # x12 = xx - shift(xx,-1)
#     # x01 = shift(xx,1) - xx
#     # x02 = shift(xx,1) - shift(xx,-1)
#
#     # d = shift(y,1)*(x12/(x01*x02))+y*(1/x12-1/x01)-shift(y,-1)*(x01/(x02*x12))
#     d[0]=y[0]*(x01[1]+x02[1])/(x01[1]*x02[1])-y[1]*x02[1]/(x01[1]*x12[1])+y[2]*x01[1]/(x02[1]*x12[1])
#     d[-1]=y[-3]*x12[-2]/(x01[-2]*x02[-2])+y[-2]*x02[-2]/(x01[-2]*x12[-2])-y[-1]*(x02[-2]+x12[-2])/(x02[-2]*x12[-2])
#
#     return d
