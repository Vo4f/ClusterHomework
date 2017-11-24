#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: id3.py
@time: 2017/11/18 15:33
@doc: ID3 Achieve
"""
import csv
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


def load_csv(file_name):
    """
    load data set
    :param file_name:
    :return: raw data in ndarray type
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data_list = list(csv.reader(f, delimiter=","))
    return data_list


def divide_data(data):
    count = {}
    for i in data:
        count[str(i)] = count.get(str(i), 0) + 1
    return count


class ID3(object):
    def __init__(self):
        self.tree = {}
        self.root_h = None

    def fit(self, file_name):
        raw = load_csv(file_name)
        data = np.asarray(raw)
        self.root_h = calc_h(data[:, -1])
        if self.root_h == 0.0:
            self.tree['root'] = data[0, -1]
            return


            # m, n = data.shape
            # root_h = calc_h(data[:, -1])
            # res_tree = {}
            # res_tmp = []
            # for arr in range(n - 2):
            #     new_arr = np.hstack((data[:, arr].reshape(data.shape[0], 1), data[:, -1].reshape(data.shape[0], 1)))
            #     res_tree[arr] = {}
            #     e = calc_e(new_arr)
            #     g = root_h - e
            #     res_tmp.append(g)
            # print(res_tmp.index(max(res_tmp)))

    def _roll(self, data, h):
        rn, cn = data.shape
        glist = []
        for attr in range(rn - 2):
            new_arr = np.hstack((data[:, attr].reshape(data.shape[0], 1), data[:, -1].reshape(data.shape[0], 1)))
            e = calc_e(new_arr)
            g = h - e
            glist.append(g)
        res_ind = glist.index(max(glist))
        self.tree[res_ind] = {}


if __name__ == '__main__':
    tree = ID3()
    tree.fit('id3-data\\sp.csv')
