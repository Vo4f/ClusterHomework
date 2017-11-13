#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: utils.py
@time: 2017/10/25 15:54
@doc: Some usefully functions
"""
import numpy as np
import calculator
import matplotlib.pyplot as plt


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


def calc_dist_array(data):
    m = data.shape[0]
    dist_array = np.ones((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            dist_array[i, j] = calculator.dist_euclidean(data[i], data[j])
    # with open('dist_array.data', 'w') as f:
    #     for i in range(m):
    #         words = ''
    #         for j in range(m):
    #             words += str(dist_array[i, j]) + ','
    #         f.writelines(words + '\n')
    return dist_array


def plot_points(data):
    x = data[:, 0]
    y = data[:, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('data')
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.scatter(x, y, c='r', marker='.')
    plt.legend('x1')
    plt.show()
