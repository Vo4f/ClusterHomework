#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: test.py
@time: 2017/10/29 20:51
@doc: 
"""
import loader
import numpy as np
import PIL.Image as Im
from kmeans import KMeans
from kmedoids import KMedoids


def data_test(dataset_name, num_cluster, methods):
    if methods == 'kmeans':
        km = KMeans(num_cluster)
    elif methods == 'kmedoids':
        km = KMedoids(num_cluster)
    else:
        km = None
    raw = loader.data_load(dataset_name)
    raw = np.delete(raw, -1, 1)
    km.fit(raw)
    label = km._labels
    class0 = label[:100]
    class1 = label[100:200]
    class2 = label[200:]
    print('---')
    print('Accuracy: ')
    print('wave 0: ' + str(max(np.sum(class0 == 0.), np.sum(class0 == 1.), np.sum(class0 == 2.)) / 100.))
    print('wave 1: ' + str(max(np.sum(class1 == 0.), np.sum(class1 == 1.), np.sum(class1 == 2.)) / 100.))
    print('wave 2: ' + str(max(np.sum(class2 == 0.), np.sum(class2 == 1.), np.sum(class2 == 2.)) / 100.))


def image_test(image_name, num_cluster, methods, gauss=False):
    img_data, row, col = loader.image_load(image_name, gauss)
    if methods == 'kmeans':
        km = KMeans(num_cluster)
    elif methods == 'kmedoids':
        km = KMedoids(num_cluster)
    else:
        km = None
    km.fit(img_data)
    label = km.labels
    label = label.reshape([row, col])
    centroid = km.centroids.astype(int)
    pic_new = Im.new("RGB", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), tuple(centroid[int(label[i][j])].tolist()))
    if gauss:
        pic_new.save(methods + '-gauss' + '.jpg')
    else:
        pic_new.save(methods + '.jpg')


def crawler_test(data_name, num_cluster):
    raw = loader.csv_load(data_name)
    words = raw[0]
    items = raw[1:]
    data = np.asarray(raw, dtype=str)
    data = np.delete(data, (0), 1)
    data = np.delete(data, (0), 0)
    data = np.asarray(data, dtype=float)
    km = KMeans(num_cluster)
    km.fit(data)
    # for ni, i in enumerate(km.centroids):
    #     for n, j in enumerate(i):
    #         if j != 0.0:
    #             print(str(ni), words[n + 1])
    # print(km.labels)
    # print(km.centroids)
    cluster = [[] for i in range(num_cluster)]
    for n, i in enumerate(km.labels):
        cluster[int(i)].append(items[n][0])
    keys = []
    for n, i in enumerate(km.centroids):
        tmp = {}
        for j, k in enumerate(i):
            tmp[words[j]] = k
        tmp = sorted(tmp.items(), key=lambda x: x[1])
        keys.append(tmp[-3:])
    for n, i in enumerate(keys):
        print(str(n)+'类', i)
    while True:
        print("请输入想要看的分类（q退出）：")
        input_number = input()
        if input_number == 'q':
            break
        print("文章列表为：")
        print(cluster[int(input_number)])


def dbscan_test(file_name, eps, min_points):
    raw = loader.mat_load(file_name)
    data = raw['a']


if __name__ == '__main__':
    # data_test('waveform012.data', 3, 'kmedoids')
    # image_test('origin.jpg', 3, 'kmeans', gauss=True)
    crawler_test('data.data', 10)
    # dbscan_test('moon.mat', 0.5, 5)
