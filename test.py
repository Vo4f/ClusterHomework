#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: test.py
@time: 2017/10/29 20:51
@doc: Kmeans的实现有Bug，经常会出现float64什么什么的错误提示，再次运行即可，多运行几次，总能正常运行
      暂时没时间debug，不影响聚类结果。
"""
import loader
import utils
import numpy as np
import PIL.Image as Im
from kmeans import KMeans
from kmedoids import KMedoids
from dbscan import DBSCAN
from id3 import ID3
from svm import SVM


def csv_test(dataset_name, num_cluster, methods):
    """
    第二次作业，Kmeas/Kmedoid，test程序
    :param dataset_name: data文件名
    :param num_cluster: 聚类数
    :param methods: 所用聚类方法
    :return: 聚类结果正确率
    """
    if methods == 'kmeans':
        km = KMeans(num_cluster)
    elif methods == 'kmedoids':
        km = KMedoids(num_cluster)
    else:
        km = None
    raw = loader.csv_load(dataset_name)
    raw = np.delete(raw, -1, 1)
    data = np.asarray(raw, dtype=float)
    km.fit(data)
    label = km.labels
    class0 = label[:100]
    class1 = label[100:200]
    class2 = label[200:]
    print('---')
    print('Accuracy: ')
    print('wave 0: ' + str(max(np.sum(class0 == 0.), np.sum(class0 == 1.), np.sum(class0 == 2.)) / 100.))
    print('wave 1: ' + str(max(np.sum(class1 == 0.), np.sum(class1 == 1.), np.sum(class1 == 2.)) / 100.))
    print('wave 2: ' + str(max(np.sum(class2 == 0.), np.sum(class2 == 1.), np.sum(class2 == 2.)) / 100.))


def image_test(image_name, num_cluster, methods, gauss=False):
    """
    第二次作业，图片聚类test程序
    :param image_name: 图片名
    :param num_cluster: 聚类数
    :param methods: 聚类方法
    :param gauss: 是否加高斯
    :return: 聚类后的图片
    """
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
    """
    第三次作业，网页聚类test程序
    :param data_name: 爬虫爬取并处理的网页数据
    :param num_cluster: 聚类数
    :return: 聚类结果，交互式
    """
    raw = loader.csv_load(data_name)
    words = raw[0]
    items = raw[1:]
    data = np.asarray(raw, dtype=str)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 0)
    data = np.asarray(data, dtype=float)
    km = KMeans(num_cluster)
    km.fit(data)
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
        print(str(n) + '类', i)
    while True:
        print("请输入想要看的分类（q退出）：")
        input_number = input()
        if input_number == 'q':
            break
        print("文章列表为：")
        print(cluster[int(input_number)])


def dbscan_test(file_name, eps, min_points, key):
    """
    第四次作业，dbscan，test程序
    :param file_name: 数据文件名
    :param eps: 核心点半径
    :param min_points: 半径所含最小点数量
    :param key: 数据文件内数据key值
    :return: 聚类后的点阵图
    """
    raw = loader.mat_load(file_name)
    # print(raw)
    data = raw[key]
    data = np.delete(data, -1, axis=1)
    # utils.plot_raw(data)
    db = DBSCAN(eps, min_points)
    db.fit(data)
    print(db.k)
    utils.plot_points(db.clusters)


def id3_test():
    t = ID3()
    t.fit('id3-data\\sp.csv')
    print('数据表示：')
    print('天气：晴-0，多云-1，有雨-2')
    print('温度：60+-0，70+-1，80+-2')
    print('湿度：60至79-0，80至89-1，90以上-2')
    print('风况：无-0，有-1')
    while True:
        data = input('请输入数据，格式为一连串数字（如0101），输入q退出：\n')
        if data == 'q':
            break
        print('结果为：')
        t.classify(data)


def svm_test():
    s = SVM()
    s.fit('svm-data\\att_faces', 'att', 20)


if __name__ == '__main__':
    # csv_test('km-data\\waveform012.data', 3, 'kmeans')
    # csv_test('km-data\\waveform012.data', 3, 'kmedoids')
    # image_test('km-data\\origin.jpg', 3, 'kmeans', gauss=True)
    # crawler_test('web-data\\web.data', 10)
    # dbscan_test('dbscan-data\\smile', 0.08, 12, 'smile')
    # dbscan_test('dbscan-data\\sizes5', 1.8, 17, 'sizes5')
    # dbscan_test('dbscan-data\\square1', 1.8, 17, 'square1')
    # dbscan_test('dbscan-data\\square4', 1.2, 26, 'b')
    # dbscan_test('dbscan-data\\spiral', 0.5, 10, 'spiral')
    # dbscan_test('dbscan-data\\moon.mat', 0.15, 12, 'a')
    # dbscan_test('dbscan-data\\long', 0.18, 10, 'long1')
    # dbscan_test('dbscan-data\\2d4c', 1.5, 20, 'a')
    # id3_test()
    svm_test()
