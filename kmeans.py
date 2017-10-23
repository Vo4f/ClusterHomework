#!/usr/bin/env python3
# encoding: utf-8
"""
@author: verf
@project: PyCharm
@file: kmeans.py
@time: 2017/10/23 8:57
@doc: Kmeans Calculater
"""
import numpy as np
import csv


class kmeans(object):
    def __init__(self, k, file):
        self.k = k
        self.file = file
        self.data = None
        self.set = None
        self.res = None
        self.ind = None
        self._centroid = None

    def file_load(self):
        with open(self.file, 'r') as f:
            raw = np.asarray(list(csv.reader(f, delimiter=",")), dtype=float)
            self.data = raw

    def cal_eucdist(self, array1, array2):
        return np.sqrt(np.sum((array1 - array2) ** 2))

    def set_rcenter(self):
        tmp = self.data
        np.random.shuffle(tmp)
        self._centroid = tmp[:self.k]
        self.set = tmp[:, :-2]
        self.res = tmp[:, -2:-1]
        self.ind = tmp[:, -1:]

    def cal_center(self):
        pass

    def _converged(self, centroid1, centroid2):
        return np.array_equal(centroid1, centroid2)

    def cal_kmeans(self):
        converge = False
        while not converge:
            for i in self.set:
                pass

    def cal_closest(self, array):
        min_dist = 10000000
        min_res = None
        for i in self._centroid:
            tmp = self.cal_eucdist(i, array)
            if tmp < min_dist:
                min_dist = tmp
                min_res = np.copy(i)
        return min_res


if __name__ == '__main__':
    km = kmeans(3, 'waveform012.data')
    km.file_load()
    km.set_rcenter()

