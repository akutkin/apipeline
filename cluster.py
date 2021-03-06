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
import logging

import matplotlib
matplotlib._log.disabled = True
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Ellipse

# from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, cKDTree

logging.basicConfig(level=logging.DEBUG)

def ra2deg(ra):
    s = np.array(ra.split(':'), dtype=float)
    if ra.startswith('-'):
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
        if (abs(px-resid_data.shape[1]) < boxsize) or (abs(py-resid_data.shape[0]) < boxsize):
            logging.debug('Skipping the edge source')
            continue

# Check if the component already in a cluster
        if clusters and any([c0.offset(c).arcmin<=cluster_radius.value*cluster_overlap for c0 in clusters]):
            # logging.debug('Already in a cluster')
            continue

        small_resid = resid_data[py-boxsize:py+boxsize, px-boxsize:px+boxsize]
        ellipse_mean, ecc, amaj, numpix = ellipses_coh(small_resid, amin=20, amax=boxsize-1, dr=1.0)

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


def voronoi_plot_2d_world(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D on "world" ax transform

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points: bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha: float, optional
        Specifies the line alpha for polygon boundaries
    point_size: float, optional
        Specifies the size of points


    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------
    Set of point:

    >>> import matplotlib.pyplot as plt
    >>> points = np.random.rand(10,2) #random

    Voronoi diagram of the points:

    >>> from scipy.spatial import Voronoi, voronoi_plot_2d
    >>> vor = Voronoi(points)

    using `voronoi_plot_2d` for visualisation:

    >>> fig = voronoi_plot_2d(vor)

    using `voronoi_plot_2d` for visualisation with enhancements:

    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                 line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()

    """
    from matplotlib.collections import LineCollection

    if ax:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:,0], vor.points[:,1], '+', markersize=point_size, transform=ax.get_transform("world"))
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:,0], vor.vertices[:,1], '.', transform=ax.get_transform("world"))

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid',
                                     transform=ax.get_transform("world")))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed',
                                     transform=ax.get_transform("world")))

    if ax:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax.figure


def write_df_voronoi(df, vor, output=None):
    output = output or 'model_clustered.txt'
    with open(output, 'w') as out:
        out.write("Format = Name,Patch,Type,Ra,Dec,I,Q,U,V,SpectralIndex,LogarithmicSI,ReferenceFrequency='1399603271.48438',MajorAxis,MinorAxis,Orientation\n")
    nn_tree = cKDTree(vor.points)
    dists, regs = nn_tree.query(np.array([df.ra, df.dec]).T)
    df['nn'] = regs
    df['dist'] = dists
    for i, p in enumerate(vor.points):
        clust_df = df.query('nn == @i').copy()
        clust_df.loc[clust_df.index,'Patch'] = 'cluster{}'.format(i+1)
        with open(output, 'a') as out:
            out.write(', cluster{}, POINT, , , , , , , , , , , ,\n'.format(i+1))
            clust_df.to_csv(out, index=False, header=False, columns=df.columns[:-4])
    return output


def voronoi_clustering(fig, ax, df, wcs, resid_data, nbright, nclusters,
                      boxsize=250, same_source_radius=5, central_region=True):
    """
    Use Voronoi clustering instead of fixed radius around sources
    """

    # logging.info('Checking {} brightest model components'.format(nbright))

    bright_df = df.sort_values('I')[::-1][:nbright][['ra', 'dec', 'I']]
    # csnrs = []
    # cfluxes = []
    # cmeasures = [] # the main value to clusterize by
    # cellipse_params = []

    # fmin, fmax = min(a.I), max(a.I)
    rms = mad(resid_data)
    # resid_mean = np.mean(resid_data)

    logging.info('Getting measures for the potential clusters...')
    clusters = []
    clusters_centers = [] #


    for ra, dec, flux in bright_df.values:
        c = SkyCoord(ra, dec, unit='deg')
        px, py = np.round(wcs.all_world2pix(c.ra, c.dec, 0)).astype(int)
# skip the edge sources
        if (abs(px-resid_data.shape[1]) < boxsize) or (abs(py-resid_data.shape[0]) < boxsize):
            # logging.debug('Skipping the edge source')
            continue
# Check if the component already in a cluster
        if clusters and any([c.separation(_).arcmin<same_source_radius for _ in clusters]):
            continue

        small_resid = resid_data[py-boxsize:py+boxsize, px-boxsize:px+boxsize]
        ellipse_mean, ecc, amaj, numpix = ellipses_coh(small_resid, amin=20, amax=boxsize-1, dr=1.0)

        if abs(ellipse_mean/rms) > 1.4:
             # rect = plt.Rectangle((px-boxsize, py-boxsize), 2*boxsize, 2*boxsize,
             #                      lw=1, color='k', fc='none', alpha=0.3)
             # ax.add_artist(rect)
             clusters_centers.append([ra, dec])
             clusters.append(c)
             print(ra, dec)

        if (isinstance(nclusters, int)) and (len(clusters_centers) >= nclusters):
            logging.debug('Max cluster number reached. Breaking...')
            break

    if isinstance(nclusters, int) and len(clusters_centers) < nclusters:
        logging.warning('Decreasing number of clusters')
        nclusters = len(clusters_centers)

    # X = np.vstack([df.ra, df.dec]).T
    # kmeans = KMeans(n_clusters=nclusters, init=clusters_centers, n_init=1, random_state=0, max_iter=1)
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X) # cluster index for each observation
    # ax.scatter(clusters_centers[:, 0], clusters_centers[:, 1], c='black', s=100, alpha=0.5, transform=ax.get_transform("world"))

    vor = Voronoi(np.array(clusters_centers))
    voronoi_plot_2d_world(vor, ax=ax, show_vertices=False)

    return vor


def main(img, resid, model, clustering_method='Voronoi', add_manual=False, nclusters=10, boxsize=250,
         nbright=80, cluster_radius=5, cluster_overlap=1.6):
    """
    clustering
    methods:
        Voronoi
        auto
        manual
    """

    imgbase = os.path.splitext(img)[0]
    output = imgbase + '-clustered.txt'

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
    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot(1,1,1, projection=wcs.celestial)
    vmin, vmax = np.percentile(image_data, 5), np.percentile(image_data, 95)
    ax.imshow(resid_data, vmin=vmin, vmax=vmax, origin='lower')#cmap='gray', vmin=2e-5, vmax=0.1)#, norm=LogNorm())

    if clustering_method.lower() == 'voronoi':

        vor = voronoi_clustering(fig, ax, df, wcs, resid_data, nbright, nclusters=nclusters)
        write_df_voronoi(df, vor, output=output)

    elif clustering_method.lower() == 'auto':
        cluster_radius = angles.Angle(cluster_radius, unit='arcmin')
        clusters = auto_clustering(fig, ax, df, wcs, resid_data, pix_arcmin_scale, nbright, cluster_radius,
                                  cluster_overlap, boxsize=boxsize, nclusters=nclusters)
        if add_manual:
            clusters_man = manual_clustering(fig, ax, wcs, pix_arcmin_scale, startnum=len(clusters)+1)
            clusters = clusters + clusters_man
        write_df(df, clusters, output=output)
    elif clustering_method.lower() == 'manual':
        clusters = manual_clustering(fig, ax, wcs, pix_arcmin_scale)
        write_df(df, clusters, output=output)

    fig.tight_layout()
    fig.savefig(imgbase+'-clustering.png')
    return output


### if __name__ == "__main__":
if __name__ == "__main__":

    base = '/kutkin/apipeline/to_validate/210109001_06-dical-'
    img = base + 'image.fits'
    resid = base + 'residual.fits'
    model = base + 'sources.txt'
    # img = sys.argv[1]
    # resid = sys.argv[2]
    # model = sys.argv[3]

    main(img, resid, model, clustering_method='Voronoi', nclusters=6)

