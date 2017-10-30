#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmedoids.py
@time: 2017/10/23 9:26
@doc: TODO waveform和图像划分都没问题，PIL没研究明白，暂时显示不出划分后的图像
"""
import numpy as np
import utils
import calculator


def calc_center(term):
    return np.mean(term, axis=0)


def calc_closet(term, centroid):
    min_dist, min_index = np.inf, -1
    for index, item in enumerate(centroid):
        dist = calculator.dist_euclidean(term, item)
        if dist < min_dist:
            min_dist = dist
            min_index = index
    return min_index


class KMeans(object):
    def __init__(self, num_clusters, init_methods='random', max_iter=300, precision=5):
        self._num_clusters = num_clusters
        self._init_methods = init_methods
        self._max_iter = max_iter
        self._precision = precision
        self._centroids = None
        self.labels_ = None
        self.centroids_ = None

    def fit(self, dataset):
        if not isinstance(dataset, np.ndarray):
            dataset = np.asarray(dataset)
        num_terms = dataset.shape[0]
        if self._init_methods == 'random':
            self._centroids = utils.init_center(dataset, self._num_clusters)
        else:
            self._centroids = self._init_methods
        converged = False
        num_iter = 0
        while not converged:
            num_iter += 1
            print("iter: " + str(num_iter))
            self.labels_ = np.zeros(num_terms)
            cluster = [[] for i in range(self._num_clusters)]
            for i in range(num_terms):
                min_index = calc_closet(dataset[i], self._centroids)
                self.labels_[i] = min_index
                cluster[min_index].append(dataset[i])
            old_centroids = [[round(j, self._precision) for j in i] for i in self._centroids]
            for index, term in enumerate(cluster):
                self._centroids[index] = calc_center(term)
            cur_centroids = [[round(j, self._precision) for j in i] for i in self._centroids]
            if utils.isconverged(old_centroids, cur_centroids) or num_iter > self._max_iter:
                converged = True
                self.centroids_ = np.copy(self._centroids)
                print('Done!')

    def predict(self, dataset):
        self.fit(dataset)
        return self.labels_
