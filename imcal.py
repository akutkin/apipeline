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


def wsclean(msin, datacolumn='DATA', outname=None, pixelsize=3, imagesize=3072, mgain=0.8, multifreq=0, autothresh=0.3,
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

    cmd = f'wsclean -name {outname} -data-column {datacolumn} -size {imagesize} {imagesize} -scale {pixelsize}asec -niter {niter} \
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


def smoothImage(imgfits) :
    """
    Smoothe an image
    """
    cmd = f'smoFits.py {imgfits}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return


def create_mask(imgfits, residfits, clipval, outname='mask.fits'):
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    cmd = f'makeNoiseMapFitsLow {imgfits} {residfits} noise.fits noiseMap.fits'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    cmd = f'makeMaskFits noiseMap.fits {outname} {clipval}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return outname


def makeNoiseImage(imgfits, residfits, low=False) :
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    if low:
      cmd = f'makeNoiseMapFitsLow {imgfits} {residfits} noiseLow.fits noiseMapLow.fits'
    else :
      cmd = f'makeNoiseMapFits {imgfits} {residfits} noise.fits noiseMap.fits'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return


def makeCombMask(ima1='noiseMap.fits', ima2='noiseMapLow.fits',
		 clip1=5, clip2=7, outname='mask.fits') :
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    cmd = f'makeCombMaskFits {ima1} {ima2} {outname} {clip1} {clip2}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return


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
           'cal.applysolution=True',
           'cal.blrange=[100,1000000]',
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
           startchan=0, nchan=0,  mode='diagonal', uvlambdamin=500, subtract=True):
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
          cal.nchan={nfreq} \
          cal.uvlambdamin={uvlambdamin} \
          steps=[cal] \
          '.format(msin=msin, msout=msout, startchan=startchan, nchan=nchan, mode=mode,
            srcdb=srcdb, solint=solint, h5out=h5out, subtract=subtract, nfreq=nfreq,
            uvlambdamin=uvlambdamin)
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
        with h5py.File(h5param, 'r') as f:
            grp = f['sol000/{}'.format(key)]
            data = grp['val'][()]
            time = grp['time'][()]
            ants = ['RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9','RTA','RTB','RTC','RTD']
            fig = plt.figure(figsize=[20, 15])
            fig.suptitle('Freq. averaged {} gain solutions'.format(key.rstrip('000')))
            for i, ant in enumerate(ants):
                ax = fig.add_subplot(4, 3, i+1)
                ax.set_title(ant)
                if key == 'amplitude000' :
                   ax.set_ylim(0,2)
                else :
                   ax.set_ylim(-180,180)
                if len(data.shape) == 5: # several directions
                    # a = ax.imshow(data[:,:,i,1,0].T, aspect='auto')
                    # plt.colorbar(a)
                    gavg = np.nanmean(data, axis=1)
                    if key == 'amplitude000' :
                      ax.plot((time-time[0])/60.0, gavg[:, i, :, 0], alpha=0.7)
                      ax.plot((time-time[0])/60.0, gavg[:, i, :, 1], alpha=0.7)
                    else :
                      ax.plot((time-time[0])/60.0, 360.0/np.pi*gavg[:, i, :, 0], alpha=0.7)
                      ax.plot((time-time[0])/60.0, 360.0/np.pi*gavg[:, i, :, 1], alpha=0.7)

                elif len(data.shape) == 4: # a single direction
                    if key == 'amplitude000' :
                      gavg = np.nanmean(data,axis=1)
#                      ax.plot((time-time[0])/3600.0, data[:, 0, i, 0], alpha=0.7)
                      ax.plot((time-time[0])/3600.0, gavg[:,  i, 0], alpha=0.7,label='XX')
                      ax.plot((time-time[0])/3600.0, gavg[:,  i, 0], alpha=0.7,label='YY')
                    else :
                      gavg = np.nanmean(data,axis=1)
#                      ax.plot((time-time[0])/3600.0, 360.0/np.pi*data[:, 0, i, 0], alpha=0.7)
#                      ax.plot((time-time[0])/3600.0, 360.0/np.pi*data[:,3 , i, 0], alpha=0.7)
                      ax.plot((time-time[0])/3600.0, 360.0/np.pi*gavg[:,  i, 0], alpha=0.7,label='XX')
                      ax.plot((time-time[0])/3600.0, 360.0/np.pi*gavg[:,  i, 1], alpha=0.7,label='YY')


                    if i == 0:
                      ax.legend(['XX','YY'])
                if i == 10:
                    ax.set_xlabel('Time (hrs)')
        return fig, ax

    if outname is not None:
        try:
            fig1, ax1 = plot_sols(h5param, 'amplitude000')
            fig1.savefig(f'{outname}_amp.png')
        except:
            logging.error('No amplitude solutions found')

        try:
            fig2, ax2 = plot_sols(h5param, 'phase000')
            fig2.savefig(f'{outname}_phase.png')
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
    mask3 = outbase + '-mask3.fits'
    mask4 = outbase + '-mask4.fits'

    nvssMod = outbase + '_nvss.sourcedb'
    model1 = outbase + '_model1.sourcedb'
    model2 = outbase + '_model2.sourcedb'
    model3 = outbase + '_model3.sourcedb'

    dical0 = outbase + '_dical0.MS'
    dical1 = outbase + '_dical1.MS'
    dical2 = outbase + '_dical2.MS'
    dical3 = outbase + '_dical3.MS'
    ddsub = outbase + '_ddsub.MS'

    h5_0 = outbase + '_dical0.h5'
    h5_1 = outbase + '_dical1.h5'
    h5_2 = outbase + '_dical2.h5'
    h5_3 = outbase + '_dical3.h5'
    h5_dd = outbase + '_ddcal.h5'


    if os.path.exists(img_ddcal+'-image.fits'):
        logging.info('The final image exists. Exiting...')
        return 0

    if cfg['nvss']['doit'] :
#    if (not os.path.exists(ms_split)) and (cfg['split1']['startchan'] or cfg['split1']['nchan']):
       ms_split = split_ms(msin, msout_path=ms_split, **cfg['split1'])

       print('------------------makesource')
       makesourcedb('nvss-model.txt', out=nvssMod)
       print('-------------nvssCal')
       dical(ms_split, nvssMod, msout=dical0, h5out=h5_0, **cfg['nvssCal'])
       view_sols(h5_0, outname=msbase+'_sols_dical0')

    else :
       ms_split = split_ms(msin, msout_path=dical0, **cfg['split1'])

    if not os.path.exists(img0 +'-image.fits') and (not os.path.exists(img0 +'-MFS-image.fits')):
        img_max = get_image_max(dical0)
        threshold = img_max/cfg['clean0']['max_over_thresh']
        threshold = max(threshold,0.001)
        wsclean(dical0, outname=img0, automask=None, save_source_list=False, multifreq=False, mgain=None,
            kwstring=f'-threshold {threshold}')
        create_mask(img0 +'-image.fits', img0 +'-residual.fits', clipval=10, outname=mask0)

# clean1
    if not os.path.exists(img1 +'-image.fits') and (not os.path.exists(img1 +'-MFS-image.fits')):
        wsclean(dical0, fitsmask=mask0, outname=img1, **cfg['clean1']) # fast shallow clean

    if not os.path.exists(model1):
        makesourcedb(img1+'-sources.txt', out=model1)
# dical1
    if not os.path.exists(dical1):
        dical1 = dical(dical0, model1, msout=dical1, h5out=h5_1, **cfg['dical1'])
        view_sols(h5_1, outname=msbase+'_sols_dical1')
# clean2
    if (not os.path.exists(img2 +'-image.fits')) and (not os.path.exists(img2 +'-MFS-image.fits')):
        wsclean(dical1, fitsmask=mask0, outname=img2, **cfg['clean2'])
        smoothImage(img2+'-residual.fits')
        makeNoiseImage(img2 +'-image.fits', img2 +'-residual.fits')
        makeNoiseImage(img2 +'-residual-smooth.fits', img2 +'-residual.fits',low=True)
        makeCombMask(outname=mask1,clip1=7,clip2=15)

    if not os.path.exists(model2):
        makesourcedb(img2+'-sources.txt', out=model2)

# dical2
    if not os.path.exists(dical2):
        dical2 = dical(dical1, model2, msout=dical2, h5out=h5_2, **cfg['dical2'])
        view_sols(h5_2, outname=msbase+'_sols_dical2')
# clean3
    if (not os.path.exists(img3 +'-image.fits')) and (not os.path.exists(img3 +'-MFS-image.fits')):
        wsclean(dical2, fitsmask=mask1, outname=img3, **cfg['clean3'])
        smoothImage(img3+'-residual.fits')
        makeNoiseImage(img3 +'-image.fits', img3 +'-residual.fits')
        makeNoiseImage(img3 +'-residual-smooth.fits', img3 +'-residual.fits',low=True)
        makeCombMask(outname=mask2,clip1=5,clip2=10)


    if not os.path.exists(model3):
        makesourcedb(img3+'-sources.txt', out=model3)

# dical3
    if not os.path.exists(dical3):
        dical3 = dical(dical2, model3, msout=dical3, h5out=h5_3, **cfg['dical3'])
        view_sols(h5_3, outname=msbase+'_sols_dical3')

# clean4
    if (not os.path.exists(img_final +'-image.fits')) and (not os.path.exists(img_final +'-MFS-image.fits')):
        wsclean(dical3, fitsmask=mask2, outname=img_final, **cfg['clean4'])
        smoothImage(img_final+'-residual.fits')
        makeNoiseImage(img_final +'-image.fits', img_final +'-residual.fits')
        makeNoiseImage(img_final +'-residual-smooth.fits', img_final +'-residual.fits',low=True)
        makeCombMask(outname=mask3,clip1=5,clip2=7)


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
        wsclean(ddsub, fitsmask=mask3,outname=img_ddsub, **cfg['clean5'])
#TAO        wsclean(ddsub,outname=img_ddsub, **cfg['clean5'])

    aomodel = bbs2model(img_final+'-sources.txt', img_final+'-model.ao')

    render(img_ddsub+'-image.fits', aomodel, out=img_ddcal+'-image.fits')

    smoothImage(img_ddsub+'-image.fits')
    makeNoiseImage(img_ddcal +'-image.fits', img_ddsub +'-residual.fits')
    makeNoiseImage(img_ddsub +'-image-smooth.fits', img_ddsub +'-residual.fits',low=True)
    makeCombMask(outname=mask4,clip1=3.5,clip2=5)

    if (not os.path.exists(img_ddsub+'-2-image.fits')):
        wsclean(ddsub, fitsmask=mask4,outname=img_ddsub+'-2', **cfg['clean5'])
#TAO        wsclean(ddsub,outname=img_ddsub, **cfg['clean5'])

    aomodel = bbs2model(img_final+'-sources.txt', img_final+'-model.ao')
    render(img_ddsub+'-2-image.fits', aomodel, out=img_ddcal+'-2-image.fits')

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

