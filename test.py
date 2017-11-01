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


def data_test(dataset_name, num_cluster):
    km = KMeans(num_cluster)
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


def image_test(image_name, num_cluster):
    img_data, row, col = loader.image_load(image_name)
    km = KMeans(num_cluster)
    km.fit(img_data)
    label = km.labels
    label = label.reshape([row, col])
    centroid = km.centroids.astype(int)
    pic_new = Im.new("RGB", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), tuple(centroid[int(label[i][j])].tolist()))
    pic_new.save('image_cluster.jpg')


if __name__ == '__main__':
    data_test('waveform012.data', 3)
    # image_test('image_test.jpg', 3)
