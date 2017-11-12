#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: utils.py
@time: 2017/10/25 15:54
@doc: Some usefully functions
"""
import numpy as np
import calculator


def init_center(data, num_clusters):
    """
    Init the raw data set, random choose centroid
    :param data: np array
    :param num_clusters: number of clusters
    :return: centroid list
    """
    tmp = data.tolist()
    np.random.shuffle(tmp)
    return tmp[:num_clusters]


def isconverged(centroid1, centroid2):
    """
    Check the KMeans is converged or not
    :param centroid1: old centroids
    :param centroid2: current centroids
    :return: the boolean
    """
    return np.array_equal(centroid1, centroid2)


def calc_dist_array(data):
    m = data.shape[0]
    dist_array = np.ones((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            dist_array[i, j] = calculator.dist_euclidean(data[i], data[j])
    return dist_array
