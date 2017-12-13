#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: ClusterHomework
@file: apriori
@time: 2017/12/11 9:44
@doc: Apriori Algorithm Achieve
"""
import numpy as np
import itertools


def init_l1(dataset):
    """
    Init l1 set
    :param dataset:
    :return:
    """
    frequent_list = {}
    for dset in dataset:
        for item in dset:
            if item in frequent_list.keys():
                frequent_list[item] += 1
            else:
                frequent_list[item] = 1
    return frequent_list


def support_count(dataset, clist):
    """
    Calculate support of given list
    :param dataset:
    :param clist:
    :return:
    """
    ln = {}
    for i in clist:
        ln[i] = 0
    for i in clist:
        for j in dataset:
            if set(i).issubset(set(j)):
                ln[i] += 1
    return ln


class Apriori(object):
    """
    Apriori Class
    """
    def __init__(self, min_support):
        self.min_support = min_support
        self.data_len = None
        self.res = None

    def fit(self, dataset):
        self.data_len = dataset.shape[0]
        ln = init_l1(dataset)
        flag = 1
        while True:
            flag += 1
            if isinstance(list(ln.keys())[0], tuple):
                cmb_n = len(list(ln.keys())[0]) + 1
            else:
                cmb_n = 2
            cn = self.generate_cn(ln, cmb_n)
            ln = support_count(dataset, cn)
            new_ln = {}
            for k, v in ln.items():
                if v >= self.min_support:
                    new_ln[k] = v
            if len(set(list(new_ln.values()))) == 1:
                self.res = new_ln
                break
            ln = new_ln

    def generate_cn(self, ln, n):
        """
        Generate new Cn set from ln-1
        :param ln:
        :param n:
        :return:
        """
        cn = list(ln.keys())
        if not isinstance(cn[0], tuple):
            set_list = list(set(cn))
        else:
            tmp = []
            for i in cn:
                tmp.extend(list(i))
            set_list = list(set(tmp))
        comb_cn = list(itertools.combinations(set_list, n))
        new_cn = []
        for i in comb_cn:
            tmp_cmb = itertools.combinations(i, n - 1)
            for j in tmp_cmb:
                if len(j) == 1:
                    cur_j = j[0]
                else:
                    cur_j = j
                if cur_j not in ln.keys() or ln[cur_j] < self.min_support:
                    continue
            new_cn.append(i)
        return new_cn


if __name__ == '__main__':
    data = np.asarray(
        [['I1', 'I2', 'I5'], ['I2', 'I4'], ['I2', 'I3'], ['I1', 'I2', 'I4'], ['I1', 'I3'], ['I2', 'I3'], ['I1', 'I3'],
         ['I1', 'I2', 'I3', 'I5'], ['I1', 'I2', 'I3']])
    ap = Apriori(2)
    ap.fit(data)
    print('找到连接')
    for k, v in ap.res.items():
        print(k)
