#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: PyCharm
@file: loader.py
@time: 2017/10/22 20:29
@doc: File Loader
"""
import numpy as np
import scipy.io as scio
import csv
import os
import PIL.Image as Im
import PIL.ImageFilter as IF


def csv_load(file_name):
    """
    load data set
    :param file_name:
    :return: raw data in ndarray type
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data_list = list(csv.reader(f, delimiter=","))
    return data_list


def image_load(file_name, gause):
    """
    load image
    :param file_name:
    :return: raw data in ndarray type, and the size of image
    """
    with open(file_name, 'rb') as f:
        data = []
        img = Im.open(f)
        if gause:
            img = img.filter(IF.GaussianBlur(radius=20))
        m, n = img.size
        for i in range(m):
            for j in range(n):
                x, y, z = img.getpixel((i, j))
                data.append([x, y, z])
    return np.asarray(data), m, n


def face_load(file_path, name, num):
    face_data = []
    face_label = []
    if name == 'att':
        for i in range(1, num + 1):
            for pit in os.listdir(file_path + "\\s" + str(i)):
                img = Im.open(file_path + "\\s" + str(i) + "\\" + pit)
                face_data.append(list(img.getdata()))
                face_label.append(i)
    return face_data, face_label


def mat_load(file_name):
    return scio.loadmat(file_name)
