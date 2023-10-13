#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:10:29 2022

@author: kutkin
"""
from astropy.coordinates import Angle
from astropy.io import fits
import numpy as np
import logging
import os
import pandas as pd


def wsrt_beam(radius) :
    """ model for old WSRT beam (r in degrees) """
    return np.cos(np.minimum(70*1.4*np.deg2rad(radius),1.0881))**6


def load_nvss(nvss_catalog_file):
    """ read NVSS catalog and return dataframe with [ra, dec, flux, err] columns """
    logging.info('Loading NVSS catalog from: %s', nvss_catalog_file)
    from zipfile import ZipFile
    if os.path.exists('nvss.csv'):
        logging.debug('NVSS catalog is already unzipped. Continuing...')
    else:
        nvsszip = ZipFile(nvss_catalog_file)
        nvsszip.extractall()
    df = pd.read_csv('nvss.csv')
    df[['flux', 'err']] /= 1e3 # to Jy
    return df


def main(img, nvsscat='/opt/nvss.csv.zip', cutoff=0.001, outname=None):
    header = fits.getheader(img)
    imsizedeg = abs(header['NAXIS1'] / 2 * header['CDELT1'])
    ra0, dec0 = header['CRVAL1'], header['CRVAL2']
    if ra0 < 0:
        ra0 += 360.0
    def scale_for_beam(df):
        raFac = np.cos(np.radians(dec0))
        radius = np.sqrt((df.ra-ra0)**2*raFac**2 + (df.dec-dec0)**2)
        beamFac = wsrt_beam(radius)
        res = df.flux*beamFac
        return res
    nvss_df = load_nvss(nvsscat)
    nvss_df = nvss_df.query('abs(ra - @ra0) <= @imsizedeg & abs(dec - @dec0) <= @imsizedeg')
    nvss_df['flux_scaled'] = nvss_df.apply(scale_for_beam, axis=1)

    if outname is None:
        outname = os.path.splitext(img)[0] + '_nvss_model.txt'

    with open(outname, 'w') as fout:
        fout.write('Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, \
                    ReferenceFrequency=\'1364100669.00262\', MajorAxis, MinorAxis, Orientation\n')
        ns = 0
        for index, row in nvss_df.iterrows():
            rah, ram, ras = Angle(row.ra, unit='deg').hms
            dd, dm, ds = Angle(row.dec, unit='deg').dms
            outstr = 's0s{},POINT,{:d}:{:d}:{:.3f},{:d}.{:d}.{:.3f},{},'.format(ns, int(rah), int(ram), ras, int(dd), int(dm), ds, row.flux_scaled)
            outstr=outstr+'[],false,1370228271.48438,,,\n'
            fout.write(outstr)
            ns += 1
    logging.info('Wrote NVSS model to %s', outname)
    return outname


if __name__ == "__main__":
    img = 'wsclean-image.fits'
    nvsscat = 'nvss.csv.zip'
    cutoff = 0.001
    main(img, cutoff=0.001, nvsscat=nvsscat, outname=None)
