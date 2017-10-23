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
    def __init__(self, k, file, l):
        self.k = k
        self.file = file
        self.l = l
        self.data = None
        self.res = None
        self._centroid = None

    def file_load(self):
        with open(self.file, 'r') as f:
            raw = np.asarray(list(csv.reader(f, delimiter=",")), dtype=float)
            self.data = raw[:self.l, :-1]
            self.res = raw[:self.l, -1:]

    def cal_eucdist(self, array1, array2):
        return np.sqrt(np.sum((array1 - array2) ** 2))

    def set_center(self):
        centroids = self.data
        np.random.shuffle(centroids)
        self._centroid = centroids[:self.k]

    def cal_closest(self, array):
        min_dist = 10000000
        min_res = None
        for i in self._centroid:
            tmp = self.cal_eucdist(i, array)
            if tmp < min_dist:
                min_dist = tmp
                min_res = i
        return i




if __name__ == '__main__':
    km = kmeans(3, 'waveform.data', 1000)
    km.file_load()
    km.set_center()
