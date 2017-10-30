#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: utils.py
@time: 2017/10/25 15:54
@doc: 常用函数集
"""
import numpy as np



def init_center(data, num_clusters):
    """
    Init the raw data set, random choose centroid
    :param data: np array
    :param num_clusters: number of clusters
    :return: centroid list
    """
    tmp = data.tolist()
    np.random.shuffle(tmp)
    return tmp[:num_clusters]


def isconverged(centroid1, centroid2):
    """
    Check the KMeans is converged or not
    :param centroid1: old centroids
    :param centroid2: current centroids
    :return: the boolean
    """
    return np.array_equal(centroid1, centroid2)


# def data_display(class_list):
#     for term in class_list:
#         num0 = 0
#         num1 = 0
#         num2 = 0
#         for i in term:
#             if i[-1:].tolist()[0] == 0.0:
#                 num0 += 1
#             if i[-1:].tolist()[0] == 1.0:
#                 num1 += 1
#             if i[-1:].tolist()[0] == 2.0:
#                 num2 += 1
#         print('---')
#         print('0: ' + str(num0))
#         print('1: ' + str(num1))
#         print('2: ' + str(num2))
#     print('---')
