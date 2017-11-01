#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: calculator.py
@time: 2017/10/25 9:59
@doc: calculator
"""
import numpy as np


def dist_euclidean(array1, array2):
    """ Calculate Euclidean Distance of two given array """
    return np.sqrt(np.sum((array1 - array2) ** 2))


def dist_manhattan(array1, array2):
    """ Calculate Manhattan Distance of two given array """
    return np.sum(np.abs(array1 - array2))
