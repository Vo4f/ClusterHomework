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
    count = divide_data(data)
    total = len(data)


def load_csv(file_name):
    """
    load data set
    :param file_name:
    :return: raw data in ndarray type
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data_list = list(csv.reader(f, delimiter=","))
    return data_list


def entropy(props):
    e = 0.0
    for s in props:
        e -= s * (log(s) / log(2))
    return e


def divide_data(data):
    count = {}
    for i in data:
        count[str(i)] = count.get(str(i), 0) + 1
    return count


class ID3(object):
    def __init__(self):
        pass

    def fit(self, file_name):
        raw = load_csv(file_name)
        data = np.asarray(raw)
        m, n = data.shape
        root_h = calc_h(data[:, -1])
        for arr in range(n - 2):
            pass


if __name__ == '__main__':
    # tree = ID3()
    # tree.fit('sp.csv')
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    b = np.asarray(a)
    c = b[:, 0].reshape(b.shape[0], 1)
    d = b[:, 2].reshape(b.shape[0], 1)
    e = np.hstack((c, d))
    print(e)
