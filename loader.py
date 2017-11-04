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
import PIL.Image as Im
import PIL.ImageFilter as IF


def data_load(file_name):
    """
    load data set
    :param file_name:
    :return: raw data in ndarray type
    """
    with open(file_name, 'r') as f:
        return np.asarray(list(csv.reader(f, delimiter=",")), dtype=float)


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
