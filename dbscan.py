#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: dbscan.py
@time: 2017/11/11 14:57
@doc: DBSCAN Achieve
"""
import utils
import numpy as np


class DBSCAN:
    """
    DBSCAN Class
    """
    def __init__(self, eps, min_points):
        """
        Parameters
        :param eps: The maximum radius for core point
        :param min_points: The number of points in radius for the point to be considered as a core point
        """
        self._eps = eps
        self._min_points = min_points
        self._dist_array = None
        self._k = 0
        self._cluster = []
        self._label = None
        self.clusters = []
        self.labels = None
        self.k = None

    def fit(self, data):
        """
        :param data: The data in type ndarray
        :return:
        """
        m = data.shape[0]
        self._dist_array = utils.calc_dist_array(data)
        core_points = []
        for i in range(m):
            if self._iscore(i):
                core_points.append(i)  # 添加符合的项的编号，类型int
        unvisited = [x for x in range(m)]  # 存储每项编号，[0, 1, 2, ...]， 编号为data数据中的顺序，类型int
        que = []
        self._label = np.zeros(m)
        while core_points:
            old_unvisited = unvisited[:]
            que.append(core_points[0])
            unvisited.remove(core_points[0])
            while que:
                first = que.pop(0)
                if first in core_points:
                    delta = [x for x in range(m) if self._dist_array[first][x] <= self._eps
                             and x in unvisited and x != first]
                    que.extend(delta)
                    unvisited = [x for x in unvisited if x not in delta]
            self._k += 1
            ck = [x for x in old_unvisited if x not in unvisited]
            for i in ck:
                self._label[i] = self._k
            self._cluster.append(ck)
            core_points = [x for x in core_points if x not in ck]
        for i in range(self._k):
            tmp = []
            for j in self._cluster[i]:
                tmp.append(data[j])
            tmp = np.asarray(tmp)
            self.clusters.append(tmp)
        self.labels = self._label
        self.k = self._k

    def _iscore(self, index):
        """
        Judge the point is core or not by it's index
        :param index: The point index
        :return: Boolean
        """
        sort_list = self._dist_array[index].tolist()
        sort_list.sort()
        res = sort_list[self._min_points - 1] <= self._eps
        return res
