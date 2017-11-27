#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: calculator.py
@time: 2017/10/25 9:59
@doc: calculator
"""
import utils
import numpy as np
from math import log


def dist_euclidean(array1, array2):
    """ Calculate Euclidean Distance of two given array """
    return np.sqrt(np.sum((array1 - array2) ** 2))


def dist_manhattan(array1, array2):
    """ Calculate Manhattan Distance of two given array """
    return np.sum(np.abs(array1 - array2))


def calc_h(data):
    """
    calculate the expectation of entropy
    :param data:
    :return:
    """
    count = utils.divide_data(data)
    total = len(data)
    h = 0
    for k, v in count.items():
        p = v / total
        h -= calc_entropy(p)
    return h


def calc_entropy(p):
    return p * (log(p) / log(2))


def calc_e(data):
    """
    calculate the average expectation of entropy
    :param data: ndarray
    :return: e
    """
    count = utils.divide_data(data[:, 0])
    m = len(count)
    t = data.shape[0]
    res = []
    for item in count.keys():
        li = np.where(data[:, 0] == item)
        tmp = []
        for i in li:
            tmp.append(data[i, :])
        r = np.vstack(tuple(tmp))
        res.append(r)
    e = 0.0
    for n, arr in enumerate(res):
        arr.astype('str')
        e += (count[str(n)] / t) * calc_h(arr[:, -1])
    return e
