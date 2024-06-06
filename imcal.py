#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imaging and self-calibration pipeline for Apertif
"""

import logging

import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
import numpy as np
import shutil
import subprocess
from subprocess import Popen as Process, TimeoutExpired, PIPE

import h5py
import pandas as pd
import glob
import yaml
import argparse


# from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

# local:
from cluster import main as cluster
from cluster import write_ds9
from nvss_cutout import main as nvss_cutout
from ghost_remove import remove_baseline_offsets, remove_ghost_from_model

from radio_beam import Beam
from radio_beam import EllipticalGaussian2DKernel
from scipy.fft import ifft2, ifftshift

import casacore.tables as ct


_POOL_TIME = 300 # SECONDS
_MAX_TIME = 1 * 3600 # SECONDS
_MAX_POOL = _MAX_TIME // _POOL_TIME


def fft_psf(bmaj, bmin, bpa, size=3073):
    SIGMA_TO_FWHM = np.sqrt(8*np.log(2))
    fmaj = size / (bmin / SIGMA_TO_FWHM) / 2 / np.pi
    fmin = size / (bmaj / SIGMA_TO_FWHM) / 2 / np.pi
    fpa = bpa + 90
    angle = np.deg2rad(90+fpa)
    fkern = EllipticalGaussian2DKernel(fmaj, fmin, angle, x_size=size, y_size=size)
    fkern.normalize('peak')
    fkern = fkern.array
    return fkern


def reconvolve_gaussian_kernel(img, old_maj, old_min, old_pa, new_maj, new_min, new_pa):
    """
    convolve image with a gaussian kernel without FFTing it
    bmaj, bmin -- in pixels,
    bpa -- in degrees from top clockwise (like in Beam)
    inverse -- use True to deconvolve.
    NOTE: yet works for square image without NaNs
    """
    size = len(img)
    imean = img.mean()
    img -= imean
    fimg = np.fft.fft2(img)
    krel = fft_psf(new_maj, new_min, new_pa, size) / fft_psf(old_maj, old_min, old_pa, size)
    fconv = fimg * ifftshift(krel)
    return ifft2(fconv).real + imean


def fits_reconvolve_psf(fitsfile, newpsf, out=None):
    """ Convolve image with deconvolution of (newpsf, oldpsf) """
    # newparams = newpsf.to_header_keywords()
    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        if currentpsf != newpsf:
            kmaj1 = (currentpsf.major.to('deg').value/hdr['CDELT2'])
            kmin1 = (currentpsf.minor.to('deg').value/hdr['CDELT2'])
            kpa1 = currentpsf.pa.to('deg').value
            kmaj2 = (newpsf.major.to('deg').value/hdr['CDELT2'])
            kmin2 = (newpsf.minor.to('deg').value/hdr['CDELT2'])
            kpa2 = newpsf.pa.to('deg').value
            norm = newpsf.to_value() / currentpsf.to_value()
            if len(hdul[0].data.shape) == 4:
                conv_data = hdul[0].data[0,0,...]
            elif len(hdul[0].data.shape) == 2:
                conv_data = hdul[0].data
            # deconvolve with the old PSF
            # conv_data = convolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1, inverse=True)
            # convolve to the new PSF
            conv_data = norm * reconvolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1,
                                                                     kmaj2, kmin2, kpa2)

            if len(hdul[0].data.shape) == 4:
                hdul[0].data[0,0,...] = conv_data
            elif len(hdul[0].data.shape) == 2:
                hdul[0].data = conv_data
            hdr = newpsf.attach_to_header(hdr)
        fits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
    return out


def modify_filename(fname, string, ext=None):
    """ name.ext --> name<string>.ext """
    fbase, fext = os.path.splitext(fname)
    if ext is not None:
        fext = ext
    return fbase + string + fext


def wsclean(msin, wsclean_bin='wsclean', datacolumn='DATA', outname=None, pixelsize=3, imagesize=3072, mgain=0.8,
            multifreq=0, autothresh=0.3,
            automask=3, niter=1000000, multiscale=False, save_source_list=True,
            clearfiles=True, clip_model_level=None,
            fitsmask=None, kwstring='', **kwargs):
    """
    wsclean
    """
    if outname is None:
        outname = os.path.splitext(msin)[0]
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


def smoothImage(imgfits, psf=30, out=None) :
    """
    Smoothe an image
    """
    if out is None:
        out = os.path.basename(imgfits.replace('.fits', '-smooth.fits'))
    return fits_reconvolve_psf(imgfits, Beam(psf*u.arcsec), out=out)


def create_mask(imgfits, residfits, clipval, outname='mask.fits'):
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    outbase = os.path.splitext(imgfits)[0]
    cmd1 = f'makeNoiseMapFitsLow {imgfits} {residfits} {outbase}_noise.fits {outbase}_noiseMap.fits'
    cmd2 = f'makeMaskFits {outbase}_noiseMap.fits {outname} {clipval}'
    logging.debug("Running command: %s", cmd1)
    subprocess.call(cmd1, shell=True)
    logging.debug("Running command: %s", cmd2)
    subprocess.call(cmd2, shell=True)
    return outname


def makeNoiseImage(imgfits, residfits, low=False) :
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    # if outbase is None:
    outbase = os.path.splitext(imgfits)[0]
    if low:
        img1, img2 = f'{outbase}_noiseLow.fits', f'{outbase}_noiseMapLow.fits'
        cmd = f'makeNoiseMapFitsLow {imgfits} {residfits} {img1} {img2}'
    else :
        img1, img2 = f'{outbase}_noise.fits', f'{outbase}_noiseMap.fits'
        cmd = f'makeNoiseMapFits {imgfits} {residfits} {img1} {img2}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return img2


def makeCombMask(img1, img2, clip1=5, clip2=7, outname=None) :
    """
    Create mask using Tom's code (e-mail on 1 Jul 2021)
    """
    if outname is None:
        outname = os.path.splitext(img1)[0] + '_mask.fits'
    cmd = f'makeCombMaskFits {img1} {img2} {outname} {clip1} {clip2}'
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return outname


def get_image_ra_dec_min_max(msin, size=3072, scale=3):
    """
    Determine image center coords, min and max values for msin
    """
    outbase = os.path.splitext(msin.rstrip('/'))[0]+'-wsclean'
    outname = outbase+'-image.fits'
    cmd = f'wsclean -name {outbase} -niter 0 -size {size} {size} -scale {scale}arcsec -gridder wgridder {msin}'
    if os.path.exists(outname):
        logging.debug('Image exists. Skipping cleaning...')
    else:
        logging.debug('Running command: %s', cmd)
        subprocess.call(cmd, shell=True)
    data = fits.getdata(outname)
    header = fits.getheader(outname)
    return outname, header['CRVAL1'], header['CRVAL2'], np.nanmin(data), np.nanmax(data)


def makesourcedb(modelfile, out=None, ):
    """ Make sourcedb file from a clustered model """
    out = out or os.path.splitext(modelfile)[0] + '.sourcedb'
    cmd = 'makesourcedb in={} out={} append=False'.format(modelfile, out)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def bbs2model(inp, out=None, ):
    """ Convert model file to AO format """
    out = out or os.path.splitext(inp)[0] + '.ao'
    cmd = 'bbs2model {} {}'.format(inp, out)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def render(bkgr, model, out=None, ):
    out = out or os.path.split(bkgr)[0] + '/restored.fits'
    cmd = 'render -a -r -t {} -o {} {}'.format(bkgr, out, model)
    logging.debug("Running command: %s", cmd)
    subprocess.call(cmd, shell=True)
    return out


def execute_dppp(args, ):
    command = ['DP3'] + args
    logging.debug('executing %s', ','.join(command))
    dppp_process = subprocess.Popen(command)
    for i in range(_MAX_POOL):
        try:
            return_code = dppp_process.wait(_POOL_TIME)
            logging.debug('DP3 process %s finished with status: %s', dppp_process.pid, return_code)
            return return_code
        except TimeoutExpired as e:
            logging.debug('DP3 process %s still running', dppp_process.pid)
            continue


def check_return_code(return_code):
    if return_code > 0:
        logging.error('An error occurred in the DPPP execution: %s', return_code)
        raise SystemExit(return_code)
    else:
        pass


def split_ms(msin_path, startchan=0, nchan=0, msout_path='', ):
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


def preflag(msin, msout=None, **kwargs):
    """
    preflag data using DP3 preflag module
    """
    if (kwargs is None) or (not any(kwargs.values())):
        logging.debug('No preflag options specified. Skipping...')
        return msin
    msout = msout or '.'
    command_args = ['steps=[preflag]',
                    f'msin={msin}',
                    f'msout={msout}',
                    'msout.overwrite=True',] + ['preflag.'+'='.join(_) for _ in kwargs.items() if _[1] is not None]
    logging.info('Flagging data (%s)', command_args)
    return_code = execute_dppp(command_args)
    logging.debug('Preflag of %s returned status code %s', msin, return_code)
    check_return_code(return_code)
    if msout == '.': msout = msin
    return msout


def dical(msin, srcdb, msout=None, h5out=None, solint=1, ntimeslots=0, startchan=0, split_nchan=0,
          mode='phaseonly', cal_nchan=0, nfreqchunks=0, uvlambdamin=500, **kwargs):
    """ direction independent calibration with DPPP """
    h5out = h5out or modify_filename(msin, f'_dical_dt{solint}_{mode}', ext='.h5')
    msout = msout or modify_filename(msin, f'_dical_dt{solint}_{mode}')
    if not cal_nchan and nfreqchunks:
        cal_nchan = ct.table(msin).getcol('DATA').shape[1]//nfreqchunks # number of freq channels in the MS
        logging.debug('Calculating Nchan for solutions, assuming %s chunks... nchan = %s', nfreqchunks, cal_nchan)
    if not solint and ntimeslots:
        solint = int((11.5 * 60 * 2) // ntimeslots)
        logging.debug('Calculating solution interval, assuming %s slots... soint = %s', ntimeslots, solint)

    command_args = [f'msin={msin}',
           f'msout={msout}',
           f'msout.overwrite=True',
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


def ddecal(msin, srcdb, msout=None, h5out=None, solint=120, ntimeslots=0, nfreq=30, nfreqchunks=6,
           startchan=0, nchan=0,  mode='diagonal', uvlambdamin=500, subtract=True, **kwargs):
    """ Perform direction dependent calibration with DPPP """
    h5out = h5out or os.path.split(msin)[0] + '/ddcal.h5'
    msbase = os.path.basename(msin).split('.')[0]
    msout = msout or '{}_{}_{}.MS'.format(msbase,mode, solint)
    if nfreqchunks:
        nfreq = ct.table(msin).getcol('DATA').shape[1]//nfreqchunks # number of freq channels in the MS
        logging.debug('Calculating Nchan for solutions, assuming %s chunks... N = %s', nfreqchunks, nfreq)
    if ntimeslots:
        solint = int((11.5 * 60 * 2) // ntimeslots)
        logging.debug('Calculating solution interval, assuming %s slots... soint = %s', ntimeslots, solint)

    cmd = 'DP3 msin={msin} msout={msout} \
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


def phase_shift(msin, new_center, msout=None, ):
    """ new_center examples: [12h31m34.5, 52d14m07.34] or [187.5deg, 52.45deg] """
    msout = msout or '.'
    cmd = "DP3 msin={msin} msout={msout} msout.overwrite=True steps=[phaseshift] \
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
            timex = (time-time[0])/3600.0
            # ants = ['RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9','RTA','RTB','RTC','RTD']
            ants = [_.decode() for _ in grp['ant'][()]]
            fig = plt.figure(figsize=[20, 15])
            fig.suptitle('Freq. averaged {} gain solutions'.format(key.rstrip('000')))
            for i, ant in enumerate(ants):
                ax = fig.add_subplot(4, 3, i+1)
                ax.set_title(ant)
                if key == 'amplitude000' :
                   ax.set_ylim(0,2)
                else :
                   ax.set_ylim(-180,180)
                gavg = np.nanmean(data, axis=1)
                if len(data.shape) == 5: # several directions
                    # a = ax.imshow(data[:,:,i,1,0].T, aspect='auto')
                    # plt.colorbar(a)
                    if key == 'amplitude000' :
                        ax.plot(timex, gavg[:, i, :, 0], alpha=0.7)
                        ax.plot(timex, gavg[:, i, :, 1], alpha=0.7)
                    else :
                        ax.plot(timex, 360.0/np.pi*gavg[:, i, :, 0], alpha=0.7)
                        ax.plot(timex, 360.0/np.pi*gavg[:, i, :, 1], alpha=0.7)

                elif len(data.shape) == 4: # a single direction
                    if key == 'amplitude000' :
                        ax.plot(timex, gavg[:,  i, 0], alpha=0.7,label='XX')
                        ax.plot(timex, gavg[:,  i, 0], alpha=0.7,label='YY')
                    else :
                        ax.plot(timex, 360.0/np.pi*gavg[:,  i, 0], alpha=0.7,label='XX')
                        ax.plot(timex, 360.0/np.pi*gavg[:,  i, 1], alpha=0.7,label='YY')
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
            fig1 = ax1 = None
            logging.debug('No amplitude solutions found')

        try:
            fig2, ax2 = plot_sols(h5param, 'phase000')
            fig2.savefig(f'{outname}_phase.png')
        except:
            fig2 = ax2 = None
            logging.debug('No phase solutions found')
    # return fig1, ax1, fig2, ax2


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


def modify_conf(cfgfile, params=None):
    """
    modify params of the cfgfile
    example params={'nvss':{'nvsscal':True, 'solint':17}, 'preflag':{'abstime':'15-Dec-2021/00:55..15-Dec-2021/01:45'}}
    """
    import ruamel.yaml as yml
    if not params:
        return
    yaml = yml.YAML()
    # yaml.preserve_quotes = True
    with open(cfgfile) as fp:
        data = yaml.load(fp)
    for key, val in params.items():
        data[key].update(val)
    with open(cfgfile, 'w') as out:
        yaml.dump(data, out)


def main(msin, steps='all', outbase=None, cfgfile=None, force=False, params=None):

    from importlib import reload
    reload(logging)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("imcal.log"),logging.StreamHandler()], force=True)


    msin = msin.rstrip('/')
    mspath = os.path.split(os.path.abspath(msin))[0]
    msbase = os.path.splitext(msin)[0]


    logging.info('Starting logger for {}'.format(__name__))
    logging.info('Processing {}'.format(msin))
    logging.info('Running steps: {}'.format(args.steps))

    t0 = Time.now()

# copy config
    local_cfgfile = msbase + '.yml'
    if cfgfile is None and not os.path.exists(local_cfgfile):
        shutil.copy2('imcal.yml', local_cfgfile)
    elif cfgfile is not None and cfgfile != local_cfgfile:
        logging.info('Copying config from: %s', cfgfile)
        shutil.copy2(cfgfile, local_cfgfile)
    # else:
        # logging.error('Check config file')
    cfgfile = local_cfgfile

    logging.info('Using config file: %s', os.path.abspath(cfgfile))

    if params:
        print(params)
        try:
            params = eval(params.replace("'", "\""))
            logging.warning('Modifying config file. Params: %s', params)
            modify_conf(cfgfile, params)
        except:
            raise Exception('Wrong params format. Example: {"nvss":{"nvsscal":True,"solint":17}}')

    os.chdir(mspath)

    with open(cfgfile) as f:
        cfg = yaml.safe_load(f)

    if steps == 'all':
        steps = ['split', 'nvss', 'preflag', 'mask', 'dical', 'ddcal']
    else:
        steps = steps.split(',')



# define file names:
    if outbase is None:
        outbase = msbase

    ms_split = msbase + '_splt.MS'

    img0 = outbase + '_0'
    img1 = outbase + '_1'
    img2 = outbase + '_2'
    img3 = outbase + '_3'
    img_dical = outbase + '-dical'
    img_ddsub_1 = outbase + '-ddsub-1'
    img_ddsub_2 = outbase + '-ddsub-2'
    img_ddcal_1 = outbase + '-ddcal-1'
    img_ddcal_2 = outbase + '-ddcal-2'
    img_ddsmsub = outbase + '-ddsmsub'
    img_ddsmo = outbase + '-ddsmo'

    mask0 = outbase + '-mask0.fits'
    mask1 = outbase + '-mask1.fits'
    mask2 = outbase + '-mask2.fits'
    mask3 = outbase + '-mask3.fits'
    mask4 = outbase + '-mask4.fits'
    mask5 = outbase + '-mask5.fits'

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



    if not force and os.path.exists(img_ddcal_2+'-image.fits'):
        logging.info('The final image exists. Exiting...')
        return 0

# get image parameters
    # if 'init' in steps:
    # if not force and os.path.exists(outbase+'-wsclean-image.fits'):
    initial_img, img_ra, img_dec, img_min, img_max = get_image_ra_dec_min_max(msin, size=cfg['clean1']['imagesize'])
    logging.info('Image: %s', initial_img)
    logging.info('Image RA, DEC: %s, %s', img_ra, img_dec)
    logging.info('Image Min, Max: %s, %s', img_min, img_max)

    if 'split' in steps:
        if os.path.exists(ms_split) and not force:
            logging.info('splitted MS exists. skipping...')
        elif (cfg['split']['startchan'] or cfg['split']['nchan']):
            msin = split_ms(msin, msout_path=ms_split, **cfg['split'])
        elif cfg['split']['crop_under1310_last_8chan']:
            logging.info('Cutting < 1310MHz and last 8 channels from MS')
            nchans = ct.table(msin).getcol('DATA').shape[1]
            if nchans == 192:
                logging.debug('Old frequency setup (192 chans). Splitting...')
                msin = split_ms(msin, msout_path=ms_split, startchan=40, nchan=192-48)
            elif nchans == 288:
                logging.debug('New frequency setup (288 chans). Splitting...')
                msin = split_ms(msin, msout_path=ms_split, startchan=20, nchan=288-28)


    if 'preflag' in steps and (not os.path.exists(outbase+'_preflagged.MS') or force) and cfg['preflag']:
        msin = preflag(msin, msout=outbase+'_preflagged.MS', **cfg['preflag'])

    if 'nvss' in steps and cfg['nvss']['nvsscal']:
        nvss_model = nvss_cutout(initial_img, nvsscat='/opt/nvss.csv.zip', cutoff=0.001)
        makesourcedb(nvss_model, out=nvssMod)
        dical0 = dical(msin, nvssMod, msout=dical0, h5out=h5_0, **cfg['nvss'])
        view_sols(h5_0, outname=msbase+'_sols_dical0')
    else:
        dical0 = msin

    if 'mask' in steps:
        if not force and (os.path.exists(img0 +'-image.fits') or (os.path.exists(img0 +'-MFS-image.fits'))):
            logging.info('mask step: Image exists, use --f to overwrite...')
        else:
            threshold = img_max/cfg['clean0']['max_over_thresh']
            threshold = max(threshold, 0.0001)
            wsclean(dical0, outname=img0, automask=None, save_source_list=False, multifreq=False, mgain=None,
                    kwstring=f'-threshold {threshold}', imagesize=cfg['clean1']['imagesize'], pixelsize=cfg['clean1']['pixelsize'])
            create_mask(img0 +'-image.fits', img0 +'-residual.fits', clipval=10, outname=mask0, )

    if 'dical' in steps:
# clean1
        if not force and (os.path.exists(img1 +'-image.fits') or (os.path.exists(img1 +'-MFS-image.fits'))):
            logging.info('dical/clean1 step: Image exists, use --f to overwrite...')
        else:
            wsclean(dical0, fitsmask=mask0, outname=img1, **cfg['clean1']) # fast shallow clean
            makesourcedb(img1+'-sources.txt', out=model1)

# dical1
        if not force and os.path.exists(dical1):
            logging.debug('dical/dical1 step: MS exists, , use --f to overwrite...')
        else:
            dical1 = dical(dical0, model1, msout=dical1, h5out=h5_1, **cfg['dical1'])
            view_sols(h5_1, outname=msbase+'_sols_dical1')
# clean2
        if not force and (os.path.exists(img2 +'-image.fits') or (os.path.exists(img2 +'-MFS-image.fits'))):
            logging.info('dical/cean2 step: Image exists, use --f to overwrite...')
        else:
            wsclean(dical1, fitsmask=mask0, outname=img2, **cfg['clean2'])
            smoothImage(img2+'-residual.fits')
            i1 = makeNoiseImage(img2 +'-image.fits', img2 +'-residual.fits', )
            i2 = makeNoiseImage(img2 +'-residual-smooth.fits', img2 +'-residual.fits', low=True, )
            makeCombMask(i1, i2, clip1=7, clip2=15, outname=mask1, )

            makesourcedb(img2+'-sources.txt', out=model2, )

# dical2
        if not force and os.path.exists(dical2):
            logging.debug('dical/dical2 step: MS exists, , use --f to overwrite...')
        else:
            dical2 = dical(dical1, model2, msout=dical2, h5out=h5_2, **cfg['dical2'])
            view_sols(h5_2, outname=msbase+'_sols_dical2')
# clean3
        if not force and (os.path.exists(img3 +'-image.fits') or (os.path.exists(img3 +'-MFS-image.fits'))):
            logging.info('dical/cean3 step: Image exists, use --f to overwrite...')
        else:
            wsclean(dical2, fitsmask=mask1, outname=img3, **cfg['clean3'])
            smoothImage(img3+'-residual.fits')
            i1 = makeNoiseImage(img3 +'-image.fits', img3 +'-residual.fits', )
            i2 = makeNoiseImage(img3 +'-residual-smooth.fits', img3 +'-residual.fits', low=True, )
            makeCombMask(i1, i2, clip1=5, clip2=10, outname=mask2,)

            makesourcedb(img3+'-sources.txt', out=model3)

# determine the solution interval for amplitude calibration (Tom's mail on 11.10.2023)

        if cfg['dical3']['solint']:
            solinterval = cfg['dical3']['solint']
        else:
            totalflux = np.nansum(fits.getdata(img3 +'-model.fits'))
            solinterval = round(max(1.0,1.0/totalflux/totalflux))*5
            logging.debug('Using optimal solution interval for DICAL3 step:, %s min', solinterval/2)

# dical3
        if not force and os.path.exists(dical3):
            logging.debug('dical/dical3 step: MS exists, use --f to overwrite...')
        else:
            cfg['dical3'].update({'solint':solinterval})
            dical3 = dical(dical2, model3, msout=dical3, h5out=h5_3, **cfg['dical3'],)
            view_sols(h5_3, outname=msbase+'_sols_dical3')

# clean4
        if not force and (os.path.exists(img_dical +'-image.fits') or (os.path.exists(img_dical +'-MFS-image.fits'))):
            logging.info('dical/cean4 step: Image exists, use --f to overwrite...')
        else:
            wsclean(dical3, fitsmask=mask2, outname=img_dical,  **cfg['clean4'])
            smoothImage(img_dical+'-residual.fits')
            i1 = makeNoiseImage(img_dical +'-image.fits', img_dical +'-residual.fits', )
            i2 = makeNoiseImage(img_dical +'-residual-smooth.fits', img_dical +'-residual.fits',low=True, )
            makeCombMask(i1, i2, clip1=5, clip2=7, outname=mask3)

    if 'ddcal' in steps:
# Cluster
        if not force and os.path.exists(img_dical +'-clustered.txt'):
            logging.info('ddcal/clustering step: cluster file exists, use --f to overwrite...')
        else:
            clustered_model = cluster(img_dical+'-image.fits', img_dical+'-residual.fits', img_dical+'-sources.txt', **cfg['cluster'])
# Makesourcedb
            clustered_sdb = makesourcedb(clustered_model, img_dical+'-clustered.sourcedb', )

# DDE calibration + peeling everything
        if not force and os.path.exists(ddsub):
            logging.debug('ddcal/ddecal step: MS exists, use --f to overwrite...')
        else:
            ddsub, h5out = ddecal(dical3, clustered_sdb, msout=ddsub, h5out=h5_dd, **cfg['ddcal'])

# Ghost removal
        if True: # maybe make optional?
            _ = remove_ghost_from_model(img_dical+'-sources.txt', fitsfile=img_dical+'-image.fits', radius=3)
            _ = remove_baseline_offsets(ddsub)

    # view the solutions and save figure
            view_sols(h5_dd, outname=msbase+'_sols_ddcal')

            if not force and os.path.exists(img_ddsub_1+'-image.fits'):
                pass
            else:
                wsclean(ddsub, fitsmask=mask3, outname=img_ddsub_1, **cfg['clean5'])
    #TAO        wsclean(ddsub,outname=img_ddsub, **cfg['clean5'])


            aomodel = bbs2model(img_dical+'-sources.txt', img_dical+'-model.ao', )
            render(img_ddsub_1+'-image.fits', aomodel, out=img_ddcal_1+'-image.fits')

            if not os.path.exists(mask4):
                smoothImage(img_ddcal_1+'-image.fits')
                i1 = makeNoiseImage(img_ddcal_1 +'-image.fits', img_ddsub_1 +'-residual.fits', )
                i2 = makeNoiseImage(img_ddcal_1 +'-image-smooth.fits', img_ddsub_1 +'-residual.fits',low=True, )
                makeCombMask(i1, i2, clip1=3.5, clip2=5, outname=mask4,)

            if not force and os.path.exists(img_ddsub_2+'-image.fits'):
                pass
            else:
                wsclean(ddsub, fitsmask=mask4, outname=img_ddsub_2, **cfg['clean5'])


            aomodel = bbs2model(img_dical+'-sources.txt', img_dical+'-model.ao', )
            render(img_ddsub_2+'-image.fits', aomodel, out=img_ddcal_2+'-image.fits', )


# create image with robust=0 (Tom's mail 22 May 2024)
        if cfg['create_robust0']:
            if not os.path.exists(img_ddsmsub+'-image.fits'):
                wsclean(ddsub, fitsmask=mask4, outname=img_ddsmsub, **cfg['clean6'])

            if not os.path.exists(mask5):
                render(img_ddsmsub+'-image.fits', aomodel, out=img_ddsmo+'-image.fits')
                smoothImage(img_ddsmsub+'-residual.fits')
                i1 = makeNoiseImage(img_ddsmo +'-image.fits', img_ddsmsub +'-residual.fits')
                i2 = makeNoiseImage(img_ddsmsub +'-residual-smooth.fits', img_ddsmsub +'-residual.fits',low=True)
                makeCombMask(i1, i2, outname=mask5, clip1=3.5, clip2=5)

            if (not os.path.exists(img_ddsmsub+'-2-image.fits')):
                wsclean(ddsub, fitsmask=mask5,outname=img_ddsmsub+'-2', **cfg['clean6'])

            render(img_ddsmsub+'-2-image.fits', aomodel, out=img_ddsmo+'-2-image.fits')




# test facet imaging:
    if 'facet' in steps:
        ds9_file = 'ddfacets.reg'
        ddvis = outbase + '_ddvis.MS'
        h5_ddvis = 'ddsols.h5'
        clustered_sdb = img_dical+'-clustered.sourcedb'
        if not os.path.exists(ddvis):
            ddvis = ddecal(dical3, clustered_sdb, msout=ddvis, subtract=False, h5out=h5_ddvis, **cfg['ddcal'])
            write_ds9(ds9_file, h5_ddvis, img_ddcal_2+'-image.fits')
        wsclean(ddvis, fitsmask=mask3, save_source_list=False, outname='img-facet', **cfg['facet_clean'],)


    extime = Time.now() - t0
    logging.info("Execution time: {:.1f} min".format(extime.to("minute").value))

    logging.info('Done')

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DDCal Inputs')
    parser.add_argument('msin', help='MS file to process')
    parser.add_argument('-c', '--config', action='store', dest='configfile', help='Config file', type=str)
    parser.add_argument('-o', '--outbase', default=None, help='output prefix', type=str)
    parser.add_argument('-s', '--steps', default='all', help='steps to run. Example: "nvss,mask,dical,ddcal"', type=str)
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite the existing files')
    parser.add_argument('-p', '--params', default=None, help='Specific config parameters', type=str)


    args = parser.parse_args()
    configfile = args.configfile or \
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imcal.yml')
    # msin = args.msin
    main(args.msin, outbase=args.outbase, steps=args.steps, cfgfile=configfile, force=args.force, params=args.params)
