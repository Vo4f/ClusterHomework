#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    def __init__(self, k, file, max_iter=50):
        self.k = k
        self.file = file
        self.max_iter = max_iter
        self.data = None
        self._centroid = None

    def file_load(self):
        with open(self.file, 'r') as f:
            raw = np.asarray(list(csv.reader(f, delimiter=",")), dtype=float)
            self.data = raw

    def cal_eucdist(self, array1, array2):
        tmp1 = array1[:-1]
        tmp2 = array2[:-1]
        return np.sqrt(np.sum((tmp1 - tmp2) ** 2))

    def set_rcenter(self):
        tmp = np.copy(self.data)
        np.random.shuffle(tmp)
        self._centroid = tmp[:self.k]

    def cal_center(self, array):
        pass

    def _converged(self, centroid1, centroid2):
        return np.array_equal(centroid1, centroid2)

    def cal_kmeans(self):
        set = np.copy(self.data)
        cent_label = []
        for i in self._centroid:
            cent_label.append(i.tolist())
        converge = False
        flag = 1
        while not converge:
            classifier = [list() for i in range(self.k)]
            print("iter:" + str(flag))
            flag += 1
            bak_cent = cent_label[:]
            old_cent = [[round(j, 10) for j in i] for i in bak_cent]
            for i in set:
                tmp = self.cal_closest(i).tolist()
                tmp_index = cent_label.index(tmp)
                classifier[tmp_index].append(i)
            for j in range(self.k):
                new_cent = np.mean(classifier[j], axis=0)
                self._centroid[j] = new_cent
                cent_label[j] = new_cent.tolist()
            cur_cent = [[round(j, 10) for j in i] for i in cent_label]
            if old_cent[:-1] == cur_cent[:-1] or flag > self.max_iter:
                converge = True
                for rl in classifier:
                    num1 = 0
                    num2 = 0
                    num3 = 0
                    for item in rl:
                        if item[-1] == 0.0:
                            num1 += 1
                        if item[-1] == 1.0:
                            num2 += 1
                        if item[-1] == 2.0:
                            num3 += 1
                    print(num1)
                    print(num2)
                    print(num3)
                    print('----')
                print("Done!")

    def cal_closest(self, array):
        tcentroid = np.copy(self._centroid)
        min_dist = 10000000
        min_res = None
        for i in tcentroid:
            tmp = self.cal_eucdist(i, array)
            if tmp < min_dist:
                min_dist = tmp
                min_res = np.copy(i)
        return min_res


if __name__ == '__main__':
    km = kmeans(3, 'waveform012.data', 500)
    km.file_load()
    km.set_rcenter()
    km.cal_kmeans()
