#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmeans.py
@time: 2017/10/23 9:26
@doc: K-Means Achieve
"""
import numpy as np
import utils
import calculator


def calc_center(clusters):
    """
    Calculate the new center of clusters
    :param clusters:
    :return: the new center in ndarray type
    """
    return np.mean(clusters, axis=0)


def calc_closest(item, centroids):
    """
    Calculate which is the closest centroid for given item
    :param item:
    :param centroids:
    :return: the index of closest centroid
    """
    min_dist, min_index = np.inf, -1
    for index, centroid in enumerate(centroids):
        dist = calculator.dist_euclidean(item, centroid)
        if dist < min_dist:
            min_dist = dist
            min_index = index
    return min_index


class KMeans(object):
    """
    KMeans class
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
        self._labels = None
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
        num_items = dataset.shape[0]
        if self._init_methods == 'random':
            self._centroids = utils.init_center(dataset, self._num_clusters)
        else:
            self._centroids = self._init_methods
        converged = False
        num_iter = 0
        while not converged:
            num_iter += 1
            print("iter: " + str(num_iter))
            self._labels = np.zeros(num_items)
            cluster = [[] for i in range(self._num_clusters)]

            # enumerate  every item in data set
            # calculate which is the closest centroid for given item
            # then mark it in labels and append the item in cluster list
            for i in range(num_items):
                min_index = calc_closest(dataset[i], self._centroids)
                self._labels[i] = min_index
                cluster[min_index].append(dataset[i])
            old_centroids = [[round(j, self._precision) for j in i] for i in self._centroids]

            # recalculate the centroids
            for index, term in enumerate(cluster):
                self._centroids[index] = calc_center(term)
            cur_centroids = [[round(j, self._precision) for j in i] for i in self._centroids]

            # check is converged or not
            if utils.isconverged(old_centroids, cur_centroids) or num_iter > self._max_iter:
                converged = True
                self.centroids = np.copy(self._centroids)
                self.labels = np.copy(self._labels)
                print('Done!')

    def predict(self, dataset):
        """
        Return the predict clusters index of given data set
        :param dataset:
        :return:
        """
        self.fit(dataset)
        return self.labels
