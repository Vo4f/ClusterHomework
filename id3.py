#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: id3.py
@time: 2017/11/18 15:33
@doc: ID3 Achieve
"""
import loader
from treelib import Node, Tree
import numpy as np
from math import log


def calc_h(data):
    count = divide_data(data)
    total = len(data)
    h = 0
    for k, v in count.items():
        p = v / total
        h -= calc_entropy(p)
    return h


def calc_entropy(p):
    return p * (log(p) / log(2))


def calc_e(data):
    count = divide_data(data[:, 0])
    m = len(count)
    t = data.shape[0]
    res = []
    for item in count.keys():
        li = np.where(data[:, 0] == item)
        tmp = []
        for i in li:
            tmp.append(data[i, :])
        r = np.vstack(tuple(tmp))
        res.append(r)
    e = 0.0
    for n, arr in enumerate(res):
        arr.astype('str')
        e += (count[str(n)] / t) * calc_h(arr[:, -1])
    return e


def divide_data(data):
    count = {}
    for i in data:
        count[str(i)] = count.get(str(i), 0) + 1
    return count


class ID3(object):
    def __init__(self):
        self.res_tree = Tree()
        self.roll_que = []
        self.attr_label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def fit(self, file_name):
        raw = loader.csv_load(file_name)
        data = np.asarray(raw)
        roll_data = RData('r', data)
        self.roll_que.append(roll_data)
        while self.roll_que:
            cdata = self.roll_que.pop(-1)
            self._roll(cdata)

    def _roll(self, data):
        key = data.key
        raw = data.raw
        rn, cn = raw.shape
        h = calc_h(raw[:, -1])
        glist = []
        for attr in range(cn - 2):
            new_arr = np.hstack((raw[:, attr].reshape(raw.shape[0], 1), raw[:, -1].reshape(raw.shape[0], 1)))
            e = calc_e(new_arr)
            print(e)
            g = h - e
            glist.append(g)
        con_attr = glist.index(max(glist))


class RData(object):
    def __init__(self, key, raw):
        self.key = key
        self.raw = raw


if __name__ == '__main__':
    tree = ID3()
    tree.fit('id3-data\\sp.csv')
