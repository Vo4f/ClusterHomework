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
import utils
import numpy as np
from calculator import calc_h, calc_e


def find_best(dataset):
    """
    find the best attribute for current dataset
    :param dataset: ndarray
    :return: the index in type int of the best attribute
    """
    rn, cn = dataset.shape
    if cn == 2:
        return 0
    h = calc_h(dataset[:, -1])
    glist = []
    for attr in range(cn - 2):
        new_arr = np.hstack(
            (dataset[:, attr].reshape(dataset.shape[0], 1), dataset[:, -1].reshape(dataset.shape[0], 1)))
        e = calc_e(new_arr)
        g = h - e
        glist.append(g)
    con_attr = glist.index(max(glist))
    return con_attr


def split_data(dataset, best_attr, value):
    """
    splite the dataset by the best attribute found before
    :param dataset: ndarray
    :param best_attr: the best attribute which found before
    :param value: the value of the best attribute
    :return: the new array
    """
    tmp = []
    for i in dataset:
        if int(i[best_attr]) == int(value):
            tmp.append(np.delete(i, best_attr))
    return np.asarray(tmp)


class ID3(object):
    def __init__(self):
        self.dtree = None
        self.attr_label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def fit(self, file_name):
        raw = loader.csv_load(file_name)
        data = np.asarray(raw)
        self.dtree = self.tree_create(data, self.attr_label)

    def tree_create(self, dataset, attr_label):
        """
        create the decision tree by recursion
        :param dataset: ndarray
        :param attr_label: the label for every attribute just for convenience
        :return: current tree
        """
        alabel = attr_label[:]
        rn, cn = dataset.shape
        end_list = dataset[:, -1].tolist()
        # if the result of data is only one, just return this result as finally result
        if end_list.count(end_list[0]) == len(end_list):
            return end_list[-1]
        # if we found all attributes and can't found the good result,
        #    we count the most of result as the finally result
        if len(attr_label) == 1:
            count = utils.divide_data(end_list)
            return sorted(count.items(), key=lambda d: d[1])[-1][0]
        best_attr = find_best(dataset)
        best_label = alabel[best_attr]
        rtree = {best_label: {}}
        del (alabel[best_attr])
        best_value = dataset[:, best_attr].tolist()
        best_class = sorted(set(best_value))
        for c in best_class:
            rtree[best_label][c] = self.tree_create(split_data(dataset, best_attr, c), alabel)
        return rtree

    def classify(self, test):
        """
        classify the test data by decision tree
        :param test: test data input in string which is a sequence of numbers
        :return: None
        """
        flag = True
        tree = self.dtree
        while flag:
            k = list(tree.keys())[0]
            ind = self.attr_label.index(k)
            x = test[ind]
            nt = None
            for i in tree[k].keys():
                if x == i:
                    nt = tree[k][i]
                    break
            if isinstance(nt, str):
                print(nt)
                flag = False
            tree = nt


if __name__ == '__main__':
    t = ID3()
    t.fit('id3-data\\sp.csv')
    t.classify('1200')
