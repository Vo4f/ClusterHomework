#!/usr/bin/env python3
# encoding: utf-8
"""
@author: verf
@project: PyCharm
@file: Kmeans.py
@time: 2017/10/23 8:57
@doc: Kmeans Calculater
"""

import numpy as np
import sklearn.datasets
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k, data, init='random', max_iter=500):
        self.k = k
        self.data = data
        self.init = init
        self.max_iter = max_iter

    def cal_eucdist(self, array1, array2):
        return np.sqrt((array1-array2) * (array1-array2).T)

    def cal_manhdist(self, array1, array2):
        return np.sum(np.abs(array1-array2))

    def sel_center(self):
        m = np.shape(self.data)[0]
        n = np.shape(self.data)[1]
        list = []
        for i in range(self.k):
            temp = random.randint(0, m)
            list.append(self.data[temp])
        return list

    def K_Means_cal(self):
        center_list = self.sel_center()
        while True:
            assort_result = self.assort_node(center_list)
            this_center_list = self.cal_center(center_list, assort_result)
            if self.compare_center(this_center_list, center_list):
                return center_list
            else:
                center_list = this_center_list

    def assort_node(self, center_list):
        templist = []
        for i in range(np.shape(self.data)[0]):
            mindis = None
            tempcenter = None
            for j in center_list:
                tempdis = self.cal_eucdist(self.data[i], j)
                if mindis == None:
                    mindis = tempdis
                    tempcenter = j
                elif mindis > tempdis:
                    mindis = tempdis
                    tempcenter = j
            templist.append((tempcenter, data[i]))
        return templist

    def cal_center(self, center_list, assort_result):
        temp_center = []
        for i in center_list:
            x = 0
            y = 0
            count = 0
            for j in assort_result:
                if str(i) == str(j[0]):
                    x += j[1][0]
                    y += j[1][1]
                    count += 1
            x = x / count
            y = y / count
            temp_center.append((x, y))
        return temp_center

    def compare_center(self, list1, list2):
        bool = False
        for i in list1:
            bool = False
            for j in list2:
                if str(i) == str(j):
                    bool = True
                    break
        return bool

    def plot(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c='g')

if __name__ == '__main__':
    iris_data = sklearn.datasets.load_iris()
    data = iris_data.data[:, 1:3]
    km = Kmeans(2, data)
    km.K_Means_cal()
    km.plot()


