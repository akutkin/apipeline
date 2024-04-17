#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:15:45 2023

@author: kutkin
"""

import sys
import casacore.tables as ct

# msin = '/home/kutkin/mnt/kutkin/lockman/test/200108171_06.MS'
# msin = sys.argv[1]
# nchans = ct.table(msin).getcol('DATA').shape[1]
# print(nchans)


def ghost_remove(msin):
    """
    remove ghost surce from visibilities
    """
    tab = ct.table(msin)
    print(tab)

ghost_remove()
