#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imaging and self-calibration pipeline for Apertif
"""

import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import subprocess
from subprocess import Popen as Process, TimeoutExpired, PIPE

import h5py
import pandas as pd
import glob
import logging
import yaml
import argparse

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

from cluster import main as cluster


_POOL_TIME = 300 # SECONDS
_MAX_TIME = 1 * 3600 # SECONDS
_MAX_POOL = _MAX_TIME // _POOL_TIME


def modify_filename(fname, string, ext=None):
    """ name.ext --> name<string>.ext """
    fbase, fext = os.path.splitext(fname)
    if ext is not None:
        fext = ext
    return fbase + string + fext


def wsclean(msin, outname=None, pixelsize=3, imagesize=3072, mgain=0.8, multifreq=0, autothresh=0.3,
            automask=3, niter=1000000, multiscale=False, save_source_list=True,
            clearfiles=True, clip_model_level=None,
            fitsmask=None, kwstring=''):
    """
    wsclean
    """
    msbase = os.path.splitext(msin)[0]
    if outname is None:
        outname = msbase
    if multiscale:
        kwstring += ' -multiscale'
    if autothresh is not None:
        kwstring += f' -auto-threshold {autothresh}'
    if automask is not None:
        kwstring += f' -auto-mask {automask}'
    if mgain:
        kwstring += f' -mgain {mgain}'
    if save_source_list:
        kwstring += ' -save-source-list'
    if multifreq:
        kwstring += f' -join-channels -channels-out {multifreq} -fit-spectral-pol 2'
    if fitsmask:
        kwstring += f' -fits-mask {fitsmask}'

    cmd = f'wsclean -name {outname} -size {imagesize} {imagesize} -scale {pixelsize}asec -niter {niter} \
            {kwstring} {msin}'
    cmd = " ".join(cmd.split())
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)

    for fname in glob.glob(outname+'*.fits'):
        newname = fname.replace('MFS-', '')
        os.rename(fname, newname)
    if clearfiles:
        todelete = glob.glob(f'{outname}-000[0-9]-*.fits') # multifreq images
        for f in todelete:
            os.remove(f)
    if save_source_list:
        remove_model_components_below_level(f'{outname}-sources.txt', clip_model_level)

    return 0


def create_mask(imgfits, residfits, clipval, outname='mask.fits'):
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    cmd = f'makeNoiseMapFits {imgfits} {residfits} noise.fits noiseMap.fits'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    cmd = f'makeMaskFits noiseMap.fits {outname} {clipval}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return outname


def get_image_max(msin):
    """
    Determine maximum image value for msin
    """
    cmd = f'wsclean -niter 0 -size 3072 3072 -scale 3arcsec -use-wgridder {msin}'
    subprocess.call(cmd, shell=True)
    return np.nanmax(fits.getdata('wsclean-image.fits'))


def makesourcedb(modelfile, out=None):
    """ Make sourcedb file from a clustered model """
    out = out or os.path.splitext(modelfile)[0] + '.sourcedb'
    cmd = 'makesourcedb in={} out={}'.format(modelfile, out)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def bbs2model(inp, out=None):
    """ Convert model file to AO format """
    out = out or os.path.splitext(inp)[0] + '.ao'
    cmd = 'bbs2model {} {}'.format(inp, out)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def render(bkgr, model, out=None):
    out = out or os.path.split(bkgr)[0] + '/restored.fits'
    cmd = 'render -a -r -t {} -o {} {}'.format(bkgr, out, model)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def execute_dppp(args):
    command = ['DPPP'] + args
    logging.debug('executing %s', ','.join(command))
    dppp_process = subprocess.Popen(command)
    for i in range(_MAX_POOL):
        try:
            return_code = dppp_process.wait(_POOL_TIME)
            logging.debug('DPPP process %s finished with status: %s', dppp_process.pid, return_code)
            return return_code
        except TimeoutExpired as e:
            logging.debug('DPPP process %s still running', dppp_process.pid)
            continue


def check_return_code(return_code):
    if return_code > 0:
        logging.error('An error occurred in the DPPP execution: %s', return_code)
        raise SystemExit(return_code)
    else:
        pass


def split_ms(msin_path, startchan, nchan=0, msout_path=''):
    """
    use casacore.tables.msutil.msconcat() to concat the new MS files
    """
    if not msout_path:
        msout_path = msin_path.replace('.MS', f'_split_{startchan}_{nchan}.MS')
    logging.debug('Splitting file %s to %s', msin_path, msout_path)
    command_args = ['steps=[]',
                    'msout.overwrite=True',
                    f'msin={msin_path}',
                    f'msin.startchan={startchan}',
                    f'msin.nchan={nchan}',
                    f'msout={msout_path}']
    return_code = execute_dppp(command_args)
    logging.debug('Split of %s returned status code %s', msin_path, return_code)
    check_return_code(return_code)
    return msout_path


def dical(msin, srcdb, msout=None, h5out=None, solint=1, startchan=0, split_nchan=0,
          mode='phaseonly', cal_nchan=0, uvlambdamin=500):
    """ direction independent calibration with DPPP """
    h5out = h5out or modify_filename(msin, f'_dical_dt{solint}_{mode}', ext='.h5')
    msout = msout or modify_filename(msin, f'_dical_dt{solint}_{mode}')
    command_args = [f'msin={msin}',
           f'msout={msout}',
           f'cal.caltype={mode}',
           f'cal.sourcedb={srcdb}',
           f'cal.solint={solint}',
           f'cal.parmdb={h5out}',
           f'cal.nchan={cal_nchan}',
           f'cal.uvlambdamin={uvlambdamin}',
           'cal.applysolution=True',
           'cal.type=gaincal',
           'steps=[cal]']
    if startchan or split_nchan:
        logging.info('Calibrating MS channels: %d - %d', startchan, split_nchan)
        command_args += [f'msin.startchan={startchan}', f'msin.nchan={split_nchan}']
    return_code = execute_dppp(command_args)
    logging.debug('DICAL returned status code %s', return_code)
    check_return_code(return_code)
    return msout

def ddecal(msin, srcdb, msout=None, h5out=None, solint=120, nfreq=30,
           startchan=0, nchan=0, minvisratio=0.6, mode='diagonal', uvlambdamin=500, subtract=True):
    """ Perform direction dependent calibration with DPPP """
    h5out = h5out or os.path.split(msin)[0] + '/ddcal.h5'
    msbase = os.path.basename(msin).split('.')[0]
    msout = msout or '{}_{}_{}.MS'.format(msbase,mode, solint)
    cmd = 'DPPP msin={msin} msout={msout} \
          msin.startchan={startchan} \
          msin.nchan={nchan} \
          msout.overwrite=true \
          cal.type=ddecal \
          cal.mode={mode} \
          cal.sourcedb={srcdb} \
          cal.solint={solint} \
          cal.h5parm={h5out} \
          cal.subtract={subtract} \
          cal.propagatesolutions=true \
          cal.propagateconvergedonly=true \
          cal.minvisratio={minvisratio} \
          cal.nchan={nfreq} \
          cal.uvlambdamin={uvlambdamin} \
          steps=[cal] \
          '.format(msin=msin, msout=msout, startchan=startchan, nchan=nchan, mode=mode,
            srcdb=srcdb, solint=solint, h5out=h5out, subtract=subtract, nfreq=nfreq,
            minvisratio=minvisratio, uvlambdamin=uvlambdamin)
    cmd = " ".join(cmd.split())
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return msout, h5out


def phase_shift(msin, new_center, msout=None):
    """ new_center examples: [12h31m34.5, 52d14m07.34] or [187.5deg, 52.45deg] """
    msout = msout or '.'
    cmd = "DPPP msin={msin} msout={msout} msout.overwrite=True steps=[phaseshift] \
           phaseshift.phasecenter={new_center}".format(**locals())
    cmd = " ".join(cmd.split())
    subprocess.call(cmd, shell=True)


def view_sols(h5param, outname=None):
    """ read and plot the gains """
    def plot_sols(h5param, key):
        print('AAA')
        figs = []
        axs = []
        with h5py.File(h5param, 'r') as f:
            grp = f['sol000/{}'.format(key)]
            data = grp['val'][()]
            time = grp['time'][()]
            ants = ['RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9','RTA','RTB','RTC','RTD']
            freq_avg_gains = np.nanmean(data, axis=1)  # average by frequency
            print(dict(f['sol000/amplitude000/val'].attrs.items())) # h5 attributes
            for ipol, pol in enumerate(['XX', 'YY']):
                fig = plt.figure(figsize=[20, 15])
                fig.suptitle('Freq. averaged {} gain solutions ({})'.format(key.rstrip('000'), pol))
                for i, ant in enumerate(ants):
                    ax = fig.add_subplot(4, 3, i+1)
                    ax.set_title(ant)
                    if key.startswith('phase'):
                        ax.plot((time-time[0])/3600.0, freq_avg_gains[:, i,...,ipol]*180.0/np.pi, alpha=0.7)
                        ax.set_ylim([-180,180])
                    else:
                        ax.plot((time-time[0])/3600.0, freq_avg_gains[:, i,...,ipol], alpha=0.7)
                    if key.startswith('amplitude'):
                        ax.set_ylim([-0.1, np.max(freq_avg_gains)])
                    if i == 0:
                        ax.legend(['c{}'.format(_) for _ in range(data.shape[-2])])
                    if i == 10:
                        ax.set_xlabel('Time (hrs)')

                figs.append(fig)
                axs.append(ax)
        return figs, axs

    try:
        (fig1, fig2), (ax1, ax2) = plot_sols(h5param, 'amplitude000')
        if outname is not None:
            fig1.savefig(f'{outname}_amp_XX.png')
            fig2.savefig(f'{outname}_amp_YY.png')
    except:
        logging.error('No amplitude solutions found')
    try:
        (fig1, fig2), (ax1, ax2) = plot_sols(h5param, 'phase000')
        if outname is not None:
            fig1.savefig(f'{outname}_phase_XX.png')
            fig2.savefig(f'{outname}_phase_YY.png')
    except:
        logging.error('No phase solutions found')


def remove_model_components_below_level(model, level=0.0, out=None):
    """
    Clip the model to be above the given level

    Parameters
    ----------
    model : STR, model file name
        the input model file name
    level : FLOAT, optional
        the threshold above which the components are kept. The default is 0.0.
    out : STR, optional
        The output model filename. The default is None (the model file will be overwritten).

    Returns
    -------
    None.
    """
    if level is None:
        return model
    out = out or model
    logging.warning('Clipping the model %s to level %f', model, level)
    df = pd.read_csv(model, skipinitialspace=True)
    new = df.query('I>@level')
    new.to_csv(out, index=False)
    return out


def main(msin, outbase=None, cfgfile='imcal.yml'):
    msin = msin.rstrip('/')
    logging.info('Processing {}'.format(msin))
    logging.info('The config file: {}'.format(cfgfile))
    with open(cfgfile) as f:
        cfg = yaml.safe_load(f)

# define file names:
    mspath = os.path.split(os.path.abspath(msin))[0]
    msbase = os.path.splitext(msin)[0]

    if outbase is None:
        outbase = msbase

    ms_split = msbase + '_splt.MS'

    img0 = outbase + '_0'
    img1 = outbase + '_1'
    img2 = outbase + '_2'
    img3 = outbase + '_3'
    img_final = outbase + '-dical'
    img_ddsub = outbase + '-ddsub'
    img_ddcal = outbase + '-ddcal'

    mask0 = outbase + '-mask0.fits'
    mask1 = outbase + '-mask1.fits'
    mask2 = outbase + '-mask2.fits'

    model1 = outbase + '_model1.sourcedb'
    model2 = outbase + '_model2.sourcedb'
    model3 = outbase + '_model3.sourcedb'

    dical1 = outbase + '_dical1.MS'
    dical2 = outbase + '_dical2.MS'
    dical3 = outbase + '_dical3.MS'
    ddsub = outbase + '_ddsub.MS'

    h5_1 = outbase + '_dical1.h5'
    h5_2 = outbase + '_dical2.h5'
    h5_3 = outbase + '_dical3.h5'
    h5_dd = outbase + '_dd.h5'


    if os.path.exists(img_ddcal+'-image.fits'):
        logging.info('The final image exists. Exiting...')
        return 0

    if (not os.path.exists(ms_split)) and (cfg['split1']['startchan'] or cfg['split1']['nchan']):
        ms_split = split_ms(msin, msout_path=ms_split, **cfg['split1'])
    else:
        ms_split = msin

# Clean + DIcal

    if not os.path.exists(img0 +'-image.fits') and (not os.path.exists(img0 +'-MFS-image.fits')):
        img_max = get_image_max(ms_split)
        threshold = img_max/cfg['clean0']['max_over_thresh']
        wsclean(ms_split, outname=img0, automask=None, save_source_list=False, multifreq=False, mgain=None,
            kwstring=f'-threshold {threshold}')
        create_mask(img0 +'-image.fits', img0 +'-residual.fits', clipval=20, outname=mask0)

# clean1
    if not os.path.exists(img1 +'-image.fits') and (not os.path.exists(img1 +'-MFS-image.fits')):
        wsclean(ms_split, fitsmask=mask0, outname=img1, **cfg['clean1']) # fast shallow clean

    if not os.path.exists(model1):
        makesourcedb(img1+'-sources.txt', out=model1)
# dical1
    if not os.path.exists(dical1):
        dical1 = dical(ms_split, model1, msout=dical1, h5out=h5_1, **cfg['dical1'])
        view_sols(h5_1, outname=msbase+'_sols_dical1')
# clean2
    if (not os.path.exists(img2 +'-image.fits')) and (not os.path.exists(img2 +'-MFS-image.fits')):
        wsclean(dical1, fitsmask=mask0, outname=img2, **cfg['clean2'])
        create_mask(img2 +'-image.fits', img2 +'-residual.fits', clipval=7, outname=mask1)

    if not os.path.exists(model2):
        makesourcedb(img2+'-sources.txt', out=model2)

# dical2
    if not os.path.exists(dical2):
        dical2 = dical(dical1, model2, msout=dical2, h5out=h5_2, **cfg['dical2'])
        view_sols(h5_2, outname=msbase+'_sols_dical2')
# clean3
    if (not os.path.exists(img3 +'-image.fits')) and (not os.path.exists(img3 +'-MFS-image.fits')):
        wsclean(dical2, fitsmask=mask1, outname=img3, **cfg['clean3'])
        create_mask(img3 +'-image.fits', img3 +'-residual.fits', clipval=5, outname=mask2)

    if not os.path.exists(model3):
        makesourcedb(img3+'-sources.txt', out=model3)

# dical3
    if not os.path.exists(dical3):
        dical3 = dical(dical2, model3, msout=dical3, h5out=h5_3, **cfg['dical3'])
        view_sols(h5_3, outname=msbase+'_sols_dical3')

# clean4
    if (not os.path.exists(img_final +'-image.fits')) and (not os.path.exists(img_final +'-MFS-image.fits')):
        wsclean(dical3, fitsmask=mask2, outname=img_final, **cfg['clean4'])


# Cluster
    if (not os.path.exists(img_final +'-clustered.txt')):
        clustered_model = cluster(img_final+'-image.fits', img_final+'-residual.fits', img_final+'-sources.txt', **cfg['cluster'])

# Makesourcedb
        clustered_sdb = makesourcedb(clustered_model, img_final+'-clustered.sourcedb')

# DDE calibration + peeling everything
    if (not os.path.exists(ddsub)):
        ddsub, h5out = ddecal(dical3, clustered_sdb, msout=ddsub, h5out=h5_dd, **cfg['ddcal'])

# view the solutions and save figure
        view_sols(h5_dd, outname=msbase+'_sols_ddcal')

    if (not os.path.exists(img_ddsub+'-image.fits')):
        wsclean(ddsub, outname=img_ddsub, **cfg['clean4'])

    aomodel = bbs2model(img_final+'-sources.txt', img_final+'-model.ao')

    render(img_ddsub+'-image.fits', aomodel, out=img_ddcal+'-image.fits')



    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Starting logger for {}'.format(__name__))
    logger = logging.getLogger(__name__)

    t0 = Time.now()
    parser = argparse.ArgumentParser(description='DDCal Inputs')
    parser.add_argument('msin', help='MS file to process')
    parser.add_argument('-c', '--config', action='store',
                        dest='configfile', help='Config file', type=str)
    parser.add_argument('-o', '--outbase', default=None, help='output prefix', type=str)

    args = parser.parse_args()
    configfile = args.configfile or \
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imcal.yml')
    msin = args.msin
    logging.info('Using config file: {}'.format(os.path.abspath(configfile)))
    main(msin, outbase=args.outbase, cfgfile=configfile)
    extime = Time.now() - t0
    print("Execution time: {:.1f} min".format(extime.to("minute").value))

