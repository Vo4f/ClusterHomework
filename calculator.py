#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: calculator.py
@time: 2017/10/25 9:59
@doc: 距离函数
"""
import numpy as np


def dist_euclidean(array1, array2):
    """
    Calculate Euclidean Distance of two array
    :param array1:
    :param array2:
    :return: int number about euclidean distance
    """
    return np.sqrt(np.sum((array1 - array2) ** 2))


def dis_manhattan(array1, array2):
    return np.sum(np.abs(array1 - array2))
