#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:15:45 2023

@author: kutkin
"""

import sys
import casacore.tables as ct
import logging
import itertools
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

import pandas as pd

# msin = '/home/kutkin/mnt/kutkin/lockman/test/200108171_06.MS'
# msin = sys.argv[1]
# nchans = ct.table(msin).getcol('DATA').shape[1]
# print(nchans)


# def get_autocorr(tab, ant, avg='time', flagged=False):
#     """ get autocorrelation data """


# def ghost_remove(msin):
#     """
#     remove ghost surce from visibilities
#     """
#     tab = ct.table(msin)
#     print(tab)


def remove_baseline_offsets(msin):
    """
    remove time-averaged baseline-based visibility offsets from measurements set
    """
    antennas = {'RT2':0,'RT3':1,'RT4':2,'RT5':3,'RT6':4,'RT7':5,'RT8':6,'RT9':7,'RTA':8,'RTB':9,'RTC':10,'RTD':11}
    antnames = ct.table(msin+'/ANTENNA').getcol('NAME')
    logging.info('ANTENNAS: %s', antnames)
    tab = ct.table(msin, readonly=False)
    nbaselines = len(list(itertools.combinations(antnames, 2)))+len(antnames)

    data = tab.getcol('DATA')
    flags = tab.getcol('FLAG')
    data[flags] = np.nan
    orig_shape = data.shape
    # flags = flags.reshape((data.shape[0]//nbaselines, nbaselines, data.shape[1], data.shape[2]))
    data = data.reshape((data.shape[0]//nbaselines, nbaselines, data.shape[1], data.shape[2]))
    for i in range(nbaselines):
        offsets = np.nanmedian(data[:,i,:,:].real, axis=0) + np.nanmedian(data[:,i,:,:].imag, axis=0)*1j
        data[:,i,:,:] -= offsets #np.nanmean(data[:,i,:,:], axis=0)
    data = data.reshape(orig_shape)
    tab.putcol('DATA', data)
    tab.close()
    return msin


def remove_ghost_from_model(modelfile, radius=2, header=None, fitsfile=None, out=None):
    """
    remove central CLEAN components from the model file
    """
    df = pd.read_csv(modelfile, skipinitialspace=True)
    c = SkyCoord(df.Ra, df.Dec.apply(lambda x: x.replace('.',':',2)), unit=(u.hourangle, u.deg))
    hdr = header or fits.getheader(fitsfile)
    c0 = SkyCoord(hdr['CRVAL1'], hdr['CRVAL2'], unit='deg')
    # print(np.where(c.separation(c0).arcsec<3))
    rem_ind = np.where(c.separation(c0).arcsec<radius)[0]
    if len(rem_ind):
        logging.warning('Removing CLEAN %s components within central %s" from %s', len(rem_ind), radius, modelfile)
        newdf = df.drop(rem_ind)
        newdf.to_csv(modelfile, index=False)
    return modelfile


if __name__=="__main__":
    remove_ghost_from_model(modelfile='/home/kutkin/mnt/hyperion/ghost_remove_test/211214003_23/211214003_23-dical-sources.txt',
                            fitsfile='/home/kutkin/mnt/hyperion/ghost_remove_test/211214003_23/211214003_23-dical-model.fits')
    # msin = '/home/kutkin/mnt/hyperion/ghost_remove_test/211214003_23_ddsub.MS'
    # remove_baseline_offsets(sys.argv[1])


    # q = ct.taql('select DATA,FLAG from $tab where ANTENNA1!=ANTENNA2')
    # data = q.getcol('DATA')

    # if not flagged:
    #     data[q.getcol('FLAG')] = np.nan
    # if avg.lower() == 'time':
    #     return abs(np.nanmean(data, axis=0))
    # elif avg.lower().startswith('freq') and len(data.shape)==3:
    #     print(data.shape)
    #     return abs(np.nanmean(data, axis=1))
    # elif avg.lower().startswith('pol') and len(data.shape)==3:
    #     return abs(np.nanmean(data, axis=2))
    # else:
    #     logging.error('Unknown average keywodr, must be time/freq/pol...')
    #     return None
