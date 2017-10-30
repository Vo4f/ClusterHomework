#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmedoids.py
@time: 2017/10/25 15:59
@doc: K-Medoids Achieve in PAM
"""


class KMedoids(object):
    """
    K-Medoids class
    """
    def __init__(self, num_clusters, init_methods='random', max_iter=300, precision=5):
        """
        Parameters
        :param num_clusters: The number of clusters.
        :param init_methods: The methods of initial centroids.
                             The default is random, if you want another, try apply what centroids you want.
        :param max_iter: The maximum number of iteration.
        :param precision: The precision of converaged.
        """
        self._num_clusters = num_clusters
        self._init_methods = init_methods
        self._max_iter = max_iter
        self._precision = precision
        self._centroids = None
        self.labels_ = None
        self.centroids_ = None

