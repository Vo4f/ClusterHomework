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
import PIL.Image as image
import calculator

def closet_center(item, centorid):
    min_dist = 10000000
    min_res = None
    for n, i in enumerate(centorid):
        dist = calculator.euclidean_dist(i, item)
        if dist < min_dist:
            min_dist = dist
            min_res = n
    return min_res


def init_center(data, num_clusters):
    """
    Init the raw data set, random choose centroid
    :param data: np array
    :param num_clusters: number of clusters
    :return: centroid list
    """
    tmp = data.tolist()
    np.random.shuffle(tmp)
    print(type(tmp))
    return tmp[:num_clusters]


def isconverged(centroid1, centroid2):
    return np.array_equal(centroid1, centroid2)


def data_display(class_list):
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


def image_display(class_list, data):
    tmp_data = np.copy(data)
    print(type(tmp_data[0]))
    print(type(data[0]))
    # for item in class_list:
    #     mean = np.mean(item, axis=0)
    #     for ind, term in enumerate(data):
    #         print()
    # print(tmp_data)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
