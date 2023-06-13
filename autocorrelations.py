#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:39:59 2023

@author: kutkin
"""

import os
import sys
import casacore.tables as ct
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import logging
import numpy as np
import argparse

from matplotlib.offsetbox import AnchoredText


def get_autocorr(tab, ant, avg='time', flagged=False):
    """ get autocorrelation data """
    q = ct.taql('select DATA,FLAG from $tab where ANTENNA1==$ant and ANTENNA2==$ant')
    data = q.getcol('DATA')
    if not flagged:
        data[q.getcol('FLAG')] = np.nan
    if avg.lower() == 'time':
        return abs(np.nanmean(data, axis=0))
    elif avg.lower().startswith('freq'):
        return abs(np.nanmean(data, axis=1))
    elif avg.lower().startswith('pol'):
        return abs(np.nanmean(data, axis=2))
    else:
        logging.error('Unknown average keywodr, must be time/freq/pol...')
        return None


def flag_ant(ms, ant, pol='YY'):
    """ flag antenna """
    poldict = {'XX':0, 'XY':1, 'YX':2, 'YY':3}
    logging.warning('overwriting MeasurementSet')
    tab = ct.table(ms, readonly=False)
    # flagtab = ct.taql('select FLAG from $tab where ANTENNA1==$ant or ANTENNA2==$ant')
    flagtab = ct.taql('select FLAG, ANTENNA1, ANTENNA2 from $tab')# where ANTENNA1==$ant or ANTENNA2==$ant')
    flagdata = flagtab.getcol('FLAG')
    ant1 = flagtab.getcol('ANTENNA1')
    ant2 = flagtab.getcol('ANTENNA2')
    flagdata[np.logical_or(ant1==ant,ant2==ant),:, poldict[pol]] = True
    tab.putcol('FLAG', flagdata)
    print(flagdata.shape, flagdata)
    tab.close()


def copy_pol(ms, ant, pol_from='XX', pol_to='YY'):
    """ copy one pol to another """
    poldict = {'XX':0, 'XY':1, 'YX':2, 'YY':3}
    logging.warning('overwriting MeasurementSet')
    tab = ct.table(ms, readonly=False)
    data = tab.getcol('DATA')
    flagdata = tab.getcol('FLAG')
    ant1 = tab.getcol('ANTENNA1')
    ant2 = tab.getcol('ANTENNA2')
    antmask = np.logical_or(ant1==ant,ant2==ant)
    data[antmask,:,poldict[pol_to]] = data[antmask,:,poldict[pol_from]]
    tab.putcol('DATA', data)
    tab.close()


def check_amplitudes(avgs):
    """
    check if some amplitude is an outlier
    """
    def outliers(arr, nstds=3):
        return np.where(abs(arr - np.mean(arr)) > nstds * np.std(arr))[0]

    avgs = np.array(avgs)
    for pol, data in zip(['XX', 'YY'], [avgs[:,0], avgs[:,3]]):
        bads = outliers(data)
        if bads:
            for bad in bads:
                logging.warning(f'BAD DATA for antenna {bad}, polarisation: {pol} (see plots)')


def plot_autocorrs(ms, flagged=False):
    """
    plot aucorrelation [time] and [freq] averages
    """
    fig1 = plt.figure(figsize=[16,10])
    fig2 = plt.figure(figsize=[16,10])
    nx=3
    ny=4
    antnames = ['RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9','RTA','RTB','RTC','RTD']
    tab = ct.table(ms)
    med_time_avgs = []
    med_freq_avgs = []
    for i in range(12):
        ant = i
        antname = antnames[i]
        avg_time = get_autocorr(tab, ant, avg='time', flagged=flagged) # TIME_AVG
        avg_freq = get_autocorr(tab, ant, avg='freq', flagged=flagged)   # FREQ_AVG
        med_time_avgs.append(np.nanmedian(avg_time, axis=0))
        med_freq_avgs.append(np.nanmedian(avg_freq, axis=0))
        for fig, res in zip ([fig1, fig2], [avg_time, avg_freq]):
            ax = fig.add_subplot(ny, nx, i+1)
            at = AnchoredText(antname, prop=dict(size=9), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            ax.plot(res[:,0], alpha=0.7) # XX
            ax.plot(res[:,3], alpha=0.7) # YY
            if i == 0:
                ax.legend(['XX', 'YY'], loc='upper right')

    fig1.text(0.5, 0.01, 'Frequency channel', fontdict={'size':11})
    fig2.text(0.5, 0.01, 'Time', fontdict={'size':11})
    for fig, name in zip([fig1, fig2], ['time_avg_autocorr', 'freq_avg_autocorr']):
        fig.text(0.01, 0.5, 'Amplitude', fontdict={'size':10}, rotation=90)
        fig.tight_layout(pad=1.5)
        figname = os.path.splitext(ms)[0].rstrip('/')
        fig.savefig(f'{figname}_{name}.png')
    check_amplitudes(med_time_avgs)
    check_amplitudes(med_freq_avgs)
    return med_time_avgs, med_freq_avgs


if __name__ == "__main__":
    res = plot_autocorrs(sys.argv[1])
    # print(res)
