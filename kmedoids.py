#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmedoids.py
@time: 2017/10/25 15:59
@doc: K-Medoids Achieve in PAM
"""
import numpy as np
import utils
import calculator


def calc_closest(item, centroids):
    """
    Calculate which is the closest centroid for given item
    :param item:
    :param centroids:
    :return: the index of closest centroid
    """
    min_dist, min_index = np.inf, -1
    for index, centroid in enumerate(centroids):
        dist = calculator.dist_manhattan(item, centroid)
        if dist < min_dist:
            min_dist = dist
            min_index = index
    return min_index, min_dist


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
        self._num_items = None
        self._clusters = None
        self._centroids = None
        self._labels = None
        self._total_cost = None
        self.centroids = None
        self.labels = None

    def fit(self, dataset):
        """
        Using K-Means to calculate clusters.
        :param dataset: data set in type ndarray
        :return: None
        """
        if not isinstance(dataset, np.ndarray):
            dataset = np.asarray(dataset)
        self._num_items = dataset.shape[0]
        if self._init_methods == 'random':
            self._centroids = utils.init_center(dataset, self._num_clusters)
        else:
            self._centroids = self._init_methods
        converged = False
        num_iter = 0
        while not converged:
            num_iter += 1
            print("iter: " + str(num_iter))
            self._clusters, self._labels, self._total_cost = self._classifier(dataset)
            for ind, cl in enumerate(self._clusters):
                for item in cl:
                    if not np.equal(item, self._centroids[ind]):
                        self._centroids[ind] = item
                        clusters, labels, total_cost = self._classifier(dataset)
                        if total_cost < self._total_cost:
                            self._total_cost = total_cost

    def _classifier(self, dataset):
        cluster = [[] for i in range(self._num_clusters)]
        label = np.zeros(self._num_items)
        total_cost = 0.0
        for i in range(self._num_items):
            min_index, min_dist = calc_closest(dataset[i], self._centroids)
            label[i] = min_index
            cluster[min_index].append(dataset[i])
            total_cost += min_dist
        return cluster, label, total_cost
