#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: utils.py
@time: 2017/10/25 15:54
@doc: 
"""
import numpy as np


def init_center(data, k):
    tmp = np.copy(data)
    np.random.shuffle(tmp)
    return tmp[:k]


def isconverged(centroid1, centroid2):
    return np.array_equal(centroid1, centroid2)


def display_res(class_list):
    for term in class_list:
        num0 = 0
        num1 = 0
        num2 = 0
        for i in term:
            if i[-1:].tolist()[0] == 0.0:
                num0 += 1
            if i[-1:].tolist()[0] == 1.0:
                num1 += 1
            if i[-1:].tolist()[0] == 2.0:
                num2 += 1
        print('---')
        print('0: ' + str(num0))
        print('1: ' + str(num1))
        print('2: ' + str(num2))
    print('---')
    print("Done!")