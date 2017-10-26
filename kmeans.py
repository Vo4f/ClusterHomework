#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmedoids.py
@time: 2017/10/23 9:26
@doc: TODO waveform和图像划分都没问题，PIL没研究明白，暂时显示不出划分后的图像
"""
import numpy as np
import calculator
import loader
import utils


def calc_new_center(term):
    return np.mean(term, axis=0)


class Kmeans(object):
    def __init__(self, k, file, type,  max_iter=50, precision=5):
        self.k = k
        self.file = file
        self.type = type
        self.max_iter = max_iter
        self.precision = precision

    def cal_kmeans(self):
        data = []
        if self.type == 'data':
            raw_data = loader.data_load(self.file)
            for l in raw_data:
                data.append(l[:-1])
        elif self.type == 'image':
            raw_data = loader.image_load(self.file)
            for l in raw_data:
                data.append(l)
        else:
            raw_data = []
        data = np.asarray(data)
        centroid = utils.init_center(data, self.k)
        converged = False
        iter_count = 0
        while not converged:
            iter_count += 1
            print('iter:' + str(iter_count))
            class_list = [[] for i in range(self.k)]
            raw_list = [[] for i in range(self.k)]
            old_centroid = [[round(j, self.precision) for j in i] for i in centroid]
            # old_centroid = centroid[:]
            for ind, item in enumerate(data):
                centroid_index = utils.closet_center(item, centroid)
                class_list[centroid_index].append(item)
                raw_list[centroid_index].append(raw_data[ind])
            for ind, term in enumerate(class_list):
                centroid[ind] = calc_new_center(term)
            cur_centroid = [[round(j, self.precision) for j in i] for i in centroid]
            # cur_centroid = centroid[:]
            if utils.isconverged(old_centroid[:-1], cur_centroid[:-1]) or iter_count > self.max_iter:
                converged = True
                if self.type == 'data':
                    utils.data_display(raw_list)
                elif self.type == 'image':
                    utils.image_display(class_list, data)
                print('Done!')


if __name__ == '__main__':
    # km = Kmeans(3, 'waveform012.data', 'data', 50, 5)
    km = Kmeans(3, 'waveform012.data', 'data', 500, 10)
    km.cal_kmeans()