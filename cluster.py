#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Clustering based on presence of artifacts around sources

Created on Tue Dec 10 20:48:31 2019

@author: kutkin
"""

import os
import sys
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, angles
from astropy.stats import median_absolute_deviation as mad
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Ellipse

# import copy

import logging
logging.basicConfig(level=logging.DEBUG)

def ra2deg(ra):
    s = np.array(ra.split(':'), dtype=float)
    if s[0] < 0.0:
        sign = -1.0
    else:
        sign = 1.0
    return sign*(abs(s[0]) + s[1]/60.0 + s[2]/3600.)*15


def dec2deg(dec):
    s = np.array(dec.split('.'), dtype=float)
    sign = np.sign(s[0])
    if len(s) == 4:
        return sign*(abs(s[0]) + s[1]/60.0 + s[2]/3600. + s[3]*10**(-len(str(s[3])))/3600.)
    elif len(s) == 3:
        return sign*(abs(s[0]) + s[1]/60.0 + s[2]/3600.)


def sep_radec(ra, dec, ra0, dec0):
    c0 = SkyCoord(ra0, dec0, unit=u.deg)
    c = SkyCoord(ra, dec, unit=u.deg)
    return c0.separation(c).arcsec


def radec(ra, dec):
    """ return SkyCoord object from ra, dec"""
    ra = ra2deg(ra)
    dec = dec2deg(dec)
    # print ra, dec
    return SkyCoord(ra, dec, unit=('deg, deg'))



class Cluster():
    """ A cluster object """
    def __init__(self, name, center, radius):
        """
        INPUT:
            name -- cluster1, ...
            center -- SkyCoord object with RA, Dec
            radius -- radius (SkyCoord Angle)
        """
        self.name, self.center, self.radius = name, center, radius

        # logging.debug(self.radius)

    # def separation(self, other):
    #     """
    #     Separation between Cluster center and ra, dec in arcmin
    #     """
    #     return self.center.separation(other.center)


    def offset(self, radec):
        """
        offset between cluster center and SkyCoord object (Angle)
        """
        return self.center.separation(radec)


    def intersects(self, other):
        """ Does it intersect with the other """
        sep = self.center.separation(other.center)
        rsum = self.radius + other.radius
        # print res, rsum
        if sep <= rsum:
            return True
        else:
            return False


# TODO merging of the clusters
    def merge(self, other, overwrite=True):
        """ merge the Cluster with the other one """
        sep = self.center.separation(other.center)
        rsum = self.radius + other.radius
        if overwrite:
            new_name = self.name
        else:
            new_name = '{}_{}'.format(self.name, other.name)
        new_center = SkyCoord((self.center.ra*self.radius/rsum + other.center.ra*other.radius/rsum),
                              (self.center.dec*self.radius/rsum + other.center.dec*other.radius/rsum))
        new_radius = max((sep, self.radius, other.radius))

        if sep > rsum:
            logging.warning('Merging the non-intersecting clusters')

        return Cluster(new_name, new_center, new_radius)


    def overplot(self, ax):
        c = self.center
        txt = self.name.lstrip('cluster')
        circle = plt.Circle((c.ra.value, c.dec.value), self.radius.deg,
                            facecolor='none', edgecolor='r', transform=ax.get_transform('world'))
        ax.text(c.ra.value, c.dec.value, '{}'.format(txt),
                ha='center', va='center',
                fontdict = {'weight':'bold','size': 13},
                transform=ax.get_transform('world'))
        ax.add_artist(circle)



def cluster_sources(df, cluster):
    radius = cluster.radius
    df['ra'] = df.Ra.apply(ra2deg)
    df['dec'] = df.Dec.apply(dec2deg)
    df['sep'] = sep_radec(df.ra, df.dec, cluster.center.ra, cluster.center.dec)/60.0
    # df['Patch']
    return df.query('sep<@radius')


def cluster_snr(df, cluster, wcs, resid_data, pix_arcmin_scale):
    """ SNR of the model sources within a cluster """
    radius=cluster.radius.arcmin
    a = cluster_sources(df, cluster)
    # signal_max = a.I.max()
    signal_sum = a.I.sum()
    px, py = np.round(wcs.all_world2pix(cluster.center.ra, cluster.center.dec, 0)).astype(int)
    # print(px, py)
    # sys.exit()
    y, x = np.mgrid[0:resid_data.shape[0], 0:resid_data.shape[1]]
    radius_pix = radius/pix_arcmin_scale
    mask = np.where((y-py)**2+(x-px)**2<=radius_pix**2)
    noise = np.std(resid_data[mask])
    return signal_sum, signal_sum/noise


def write_df(df, clusters, output=None):
    with open(output, 'w') as out:
        out.write("Format = Name,Patch,Type,Ra,Dec,I,Q,U,V,SpectralIndex,LogarithmicSI,ReferenceFrequency='1399603271.48438',MajorAxis,MinorAxis,Orientation\n")
    if not clusters:
        logging.error('No clusters')
        return -1
    for cluster in clusters:
        df['sep'] = sep_radec(df.ra, df.dec, cluster.center.ra, cluster.center.dec)
        clust = df.query('sep <= @cluster.radius.arcsec')
        clust.loc[clust.index,'Patch'] = cluster.name
        df.loc[clust.index, 'Patch'] = cluster.name
        clusternum = cluster.name.lstrip('cluster')
        with open(output, 'a') as out:
            out.write(', {}, POINT, , , , , , , , , , , ,\n'.format(cluster.name))
            clust.to_csv(out, index=False, header=False, columns=df.columns[:-3])

    clust = df.query('Patch == "cluster0"')
    clusternum = int(clusternum) + 1
    restname = 'cluster' + str(clusternum)
    clust.loc[clust.index,'Patch'] = restname
    df.loc[clust.index, 'Patch'] = restname
    with open(output, 'a') as out:
        out.write(', {}, POINT, , , , , , , , , , , ,\n'.format(restname))
        clust.to_csv(out, index=False, header=False, columns=df.columns[:-3])

    return 0


def radial_profile(ra, dec, resid_img):
    window = 180
    step = 10
    # sampling=200
    initial_radius = 15
    final_radius = window
    rads = np.arange(initial_radius, final_radius, step)
    with fits.open(resid_img) as f:
        wcs = WCS(f[0].header).celestial
        pix_arcmin_scale = f[0].header['CDELT2']*60
        resid_data = f[0].data[0,0,...]
    c = radec(ra, dec)
    px, py = wcs.all_world2pix(c.ra, c.dec, 0)
    px, py = int(round(py)), int(round(px))
    res = np.zeros_like(rads, dtype=float)
    for ind, rad in enumerate(rads):
        sampling = int(1000 * rad / final_radius)
        for angle in np.linspace(0, 2*np.pi, sampling):
            x = int(rad * np.cos(angle))
            y = int(rad * np.sin(angle))
            d = resid_data[y+py:y+py+step, x+px:x+px+step]
            res[ind] += np.nanmean(abs(d)) / sampling
    return rads, res


def sector_max(ra, dec, resid_img, ax, nsectors=6):
    r0 = 10
    r1 = 200 # pixels
    with fits.open(resid_img) as f:
        wcs = WCS(f[0].header).celestial
        img_size = f[0].header['NAXIS1']
        pix_arcmin_scale = f[0].header['CDELT2']*60
        resid_data = f[0].data[0,0,...]

    mad_std = mad(resid_data)

    c = radec(ra, dec)
    px, py = wcs.all_world2pix(c.ra, c.dec, 0)
    px, py = int(round(px)), int(round(py))

    y, x = np.mgrid[0:img_size,0:img_size]
    x = x-px
    y = y-py
    radcond = np.logical_and(np.hypot(x,y)>r0, np.hypot(x,y)<r1)
    sectors = np.linspace(-np.pi, np.pi, nsectors)
    result = []


    # fig = plt.figure(figsize=[10,10])
    # ax = fig.add_subplot(1,1,1, projection=wcs.celestial)
    # vmin, vmax = np.percentile(resid_data, 5), np.percentile(resid_data, 95)
    # ax.imshow(resid_data, vmin=vmin, vmax=vmax, origin='lower')#cmap='gray', vmin=2e-5, vmax=0.1)#, norm=LogNorm())

    for i in range(nsectors-1):
        ang1, ang2 = sectors[i], sectors[i+1]
        # angcond = np.logical_and(x*np.sin(ang1)/np.cos(ang1) < y, y <= x*np.sin(ang2)/np.cos(ang2))
        coef = np.arctan2(y,x)
        angcond = np.logical_and(ang1 < coef, coef < ang2)
        cond = np.logical_and(radcond, angcond)
        res = np.nanmax(resid_data[cond]) / mad_std
        result.append(res)

        cond2 = np.logical_and(cond, resid_data==max(resid_data[cond]))
        yy, xx = np.argwhere(cond2)[0]
        ra, dec = wcs.all_pix2world(xx, yy, 0)
        # print yy, xx
        if res > 10:
            ax.plot(ra, dec, '.r', transform=ax.get_transform('world'))

        # tmp = np.argwhere(cond)
        # for p in tmp:
        #     ax.plot(p[1], p[0], '.k')
        # print ang1, ang2, len(cond[cond])

    return np.array(result)



def ellipses_coh(img, x0=None, y0=None, dr=None, amin=20, amax=100):
    """
    Find an ellipse ring with the highest absolute mean pixels value.
    Return: max of abs of mean of the pixels within various ellipses,
    minor_axis/major_axis, major_axis, number of pixels within the ellipse
    """

    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x0 = x0 or int(img.shape[1]/2)
    y0 = y0 or int(img.shape[0]/2)
    y, x = y-y0, x-x0

    eccs = np.linspace(0.6, 1.0, 10)
    arange = range(amin, amax)

    # drmin = 1.0
    # drmax = 40.0
    # OldRange = float(amax - amin)
    # NewRange = float(drmax - drmin)


    res = np.zeros((len(eccs), len(arange)))
    for i, ecc in enumerate(eccs):
        for j, a in enumerate(arange):
            # dr = (((a - amin) * NewRange) / OldRange) + drmin
            b = a*ecc
            cond = np.logical_and(x**2*(a+dr)**2 + y**2*(b+dr)**2 < (a+dr)**2*(b+dr)**2, x**2*a**2 + y**2*b**2 >= a**2*b**2)
            if len(cond[cond])>=50:
                # res[i, j] = abs(sum(img[cond]))/len(img[cond])
                res[i, j] = abs(np.nanmean(img[cond]))
            else:
                logging.warning('{:d} pixels for dr={:.1f} a={:0.1f}, e={:.1f}'.format(len(cond[cond]), dr, a, ecc))
                res[i, j] = 0.0


    imax, jmax = np.argwhere(res==res.max())[0]
    eccmax = eccs[imax]
    amax = arange[jmax]
    bmax = amax*eccmax
    cond = np.logical_and(x**2*(amax+dr)**2 + y**2*(bmax+dr)**2 < (amax+dr)**2*(bmax+dr)**2,
                          x**2*amax**2 + y**2*bmax**2 >= amax**2*bmax**2)
    # logging.debug('Ellipse size: {:d} pixels'.format(len(img[cond])))
    return res.max(), eccmax, amax, len(img[cond])


def manual_clustering(fig, ax, wcs, pix_arcmin_scale, startnum=1):

    def get_cluster(cen, rad, name):
        xc, yc = cen
        x, y = rad
        radius_pix = np.hypot(x-xc, y-yc)
        radius = angles.Angle(radius_pix * pix_arcmin_scale, unit='arcmin')
        ra, dec = wcs.all_pix2world(xc, yc, 0)
        center = SkyCoord(ra, dec, unit='deg')
        logging.info("Cluster {} at {} of {} radius".format(name, center, radius))
        return Cluster(name, center, radius)

    do = True
    i = startnum
    clusters = []
    while do:
        logging.info('Select center and radius for the cluster. \
                     Then press left button to continue or middle to skip. \
                     Right button -- to cancel the last selection.')
        inp = fig.ginput(3, timeout=-1)
        cen, rad = inp[:2]
        if len(inp) == 2:
            do = False
        cluster = get_cluster(cen, rad, 'cluster{}'.format(i))
        cluster.overplot(ax)
        clusters.append(cluster)
        i += 1

    return clusters


def auto_clustering(fig, ax, df, wcs, resid_data, pix_arcmin_scale, nbright,
                    cluster_radius, cluster_overlap, boxsize=250, nclusters=5):

    a = df.sort_values('I')[::-1][:nbright][['Ra', 'Dec', 'I']]
    clusters = [] #
    csnrs = []
    cfluxes = []
    cmeasures = [] # the main value to clusterize by
    cellipse_params = []

    fmin, fmax = min(a.I), max(a.I)

    rms = mad(resid_data)
    resid_mean = np.mean(resid_data)

    if nclusters == 'auto':
        logging.info('Number of clusters will be determined automatically')
    else:
        logging.info('Maximum number of clusters is {}'.format(nclusters))

    logging.info('Getting measures for the potential clusters...')
    src_index = 1
    cluster_index = 1

    for ra, dec, flux in a.values:
        c = radec(ra, dec)
        px, py = np.round(wcs.all_world2pix(c.ra, c.dec, 0)).astype(int)
        # print(px, py)
        # print(src_index, ra, dec, px, py, flux)
        src_index += 1

# skip the edge sources
        if (abs(px-resid_data.shape[1]) < boxsize)or (abs(py-resid_data.shape[0]) < boxsize):
            logging.debug('Skipping the edge source')
            continue

# Check if the component already in a cluster
        if clusters and any([c0.offset(c).arcmin<=cluster_radius.value*cluster_overlap for c0 in clusters]):
            # logging.debug('Already in a cluster')
            continue

        small_resid = resid_data[py-boxsize:py+boxsize, px-boxsize:px+boxsize]
        ellipse_mean, ecc, amaj, numpix = ellipses_coh(small_resid, amin=20, amax=boxsize-1, dr=1.0)
        # print ellipse_mean, ecc, amaj, numpix



        # if abs(ring_mean/resid_mean) > 1.7e5:
        # if abs(ring_mean/small_resid_mean) > 1e4:

        if nclusters=='auto':
            if abs(ellipse_mean/rms) > 1.4:
                 rect = plt.Rectangle((px-boxsize, py-boxsize), 2*boxsize, 2*boxsize,
                                      lw=2, color='k', fc='none')
                 ellipse = Ellipse(xy=(px,py), width=2*amaj*ecc, height=2*amaj,
                     angle=0, lw=3, color='gray', fc='none', alpha=0.5)
                 ax.add_artist(rect)
                 ax.add_artist(ellipse)
                 cluster_name = 'cluster{}'.format(cluster_index)
                 cluster = Cluster(cluster_name, c, cluster_radius)
                 csnr = cluster_snr(df, cluster, wcs, resid_data, pix_arcmin_scale)[1]
                 if csnr < 100: # skip clusters with low SNR
                     logging.debug('Skipping low SNR cluster at {}'.format(cluster.center))
                     continue
                 clusters.append(cluster)
                 cluster.overplot(ax)
                 print(cluster_name, ra, dec, csnr, boxsize)
                 cluster_index += 1
        else:
            cluster_name = 'cluster{}'.format(src_index)
            cluster = Cluster(cluster_name, c, cluster_radius)
            cflux, csnr = cluster_snr(df, cluster, wcs, resid_data, pix_arcmin_scale)
            clusters.append(cluster)
            cfluxes.append(cflux)
            csnrs.append(csnr)
            cmeasures.append(abs(ellipse_mean/resid_mean))
            cellipse_params.append([amaj, ecc, numpix])

    if nclusters == 'auto':
        return clusters
    else:
        indexes = np.argsort(cmeasures)[::-1][:nclusters]

        final_clusters = []
        logging.info('Picking {} clusters'.format(nclusters))

        for i in indexes:
            cmeasure = cmeasures[i]
            cluster = clusters[i]
            amaj, ecc, npix = cellipse_params[i]
            csnr = csnrs[i]
            cflux = cfluxes[i]

            if csnr < 100: # skip clusters with low SNR
                logging.debug('Skipping low SNR cluster at {}'.format(cluster.center))
                continue

            cluster.name = 'cluster{}'.format(cluster_index)
            print(cluster.name, ra, dec, csnr, cmeasure)

            px, py = wcs.all_world2pix(cluster.center.ra, cluster.center.dec, 0)
            px, py = int(px), int(py)

            rect = plt.Rectangle((px-boxsize, py-boxsize), 2*boxsize, 2*boxsize,
                                 lw=2, color='k', fc='none')
            ellipse = Ellipse(xy=(px,py), width=2*amaj*ecc, height=2*amaj,
                angle=0, lw=3, color='gray', fc='none', alpha=0.5)
            ax.add_artist(rect)
            ax.add_artist(ellipse)

            final_clusters.append(cluster)
            cluster.overplot(ax)
            cluster_index += 1

        return final_clusters


def main(img, resid, model, auto=True, add_manual=False, nclusters=5, boxsize=250,
         nbright=80, cluster_radius=5, cluster_overlap=1.6):

    path = os.path.split(os.path.abspath(img))[0]
    output = os.path.join(path, 'clustered.txt')

    df = pd.read_csv(model, skipinitialspace=True)
    df['ra'] = df.Ra.apply(ra2deg)
    df['dec'] = df.Dec.apply(dec2deg)

    df.insert(1, 'Patch', 'cluster0')
    df.insert(6, 'Q', 0)
    df.insert(7, 'U', 0)
    df.insert(8, 'V', 0)

    image_data = fits.getdata(img)[0,0,...]
    resid_data = fits.getdata(resid)[0,0,...]
    with fits.open(img) as f:
        wcs = WCS(f[0].header).celestial
        pix_arcmin_scale = f[0].header['CDELT2']*60
        # racen = f[0].header['CRVAL1']
        # deccen = f[0].header['CRVAL2']
    cluster_radius = angles.Angle(cluster_radius, unit='arcmin')
    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot(1,1,1, projection=wcs.celestial)
    vmin, vmax = np.percentile(image_data, 5), np.percentile(image_data, 95)
    ax.imshow(resid_data, vmin=vmin, vmax=vmax, origin='lower')#cmap='gray', vmin=2e-5, vmax=0.1)#, norm=LogNorm())
    if auto:
        clusters = auto_clustering(fig, ax, df, wcs, resid_data, pix_arcmin_scale, nbright, cluster_radius,
                                  cluster_overlap, boxsize=boxsize, nclusters=nclusters)
        if add_manual:
            clusters_man = manual_clustering(fig, ax, wcs, pix_arcmin_scale, startnum=len(clusters)+1)
            clusters = clusters + clusters_man
    else:
        clusters = manual_clustering(fig, ax, wcs, pix_arcmin_scale)

    if clusters:
        write_df(df, clusters, output=output)
    fig.tight_layout()
    fig.savefig(path+'/clustering.png')


### if __name__ == "__main__":
if __name__ == "__main__":

    # base = '/home/kutkin/apertif/clustering/191209026/01/dical2-'
    # base = '/home/kutkin/apertif/clustering/191209026/15/dical2-'
    # base = '/home/kutkin/apertif/clustering/191010042/00/dical2-'
    # base = '/home/kutkin/apertif/clustering/191006041_21/dical2-'
    # base = '/home/kutkin/apertif/clustering/191010042_25/dical-'
    # base = '/home/kutkin/apertif/clustering/191209026/10/dical2-'
    base = '/home/kutkin/apertif/clustering/191010041/23/dical2-'
    # base = '/home/kutkin/apertif/clustering/190915041/25/dical2-'

    img = base + 'image.fits'
    resid = base + 'residual.fits'
    model = base + 'sources.txt'
    # img = sys.argv[1]
    # resid = sys.argv[2]
    # model = sys.argv[3]

    main(img, resid, model, auto=True, nclusters='auto')

