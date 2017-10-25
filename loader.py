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
import csv
import PIL.Image as image


def data_load(file_name):
    with open(file_name, 'r') as f:
        raw = np.asarray(list(csv.reader(f, delimiter=",")), dtype=float)
    return raw


def image_load(file_name):
    with open(file_name, 'rb') as f:
        data = []
        im = image.open(f)
        m, n = im.size
        print(m, n)
        for i in range(m):
            for j in range(n):
                tmp = []
                x, y, z = im.getpixel((i, j))
                tmp.append(x / 256.0)
                tmp.append(y / 256.0)
                tmp.append(z / 256.0)
                data.append(tmp)
    return np.mat(data)
