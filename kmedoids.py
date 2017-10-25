#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: kmedoids.py
@time: 2017/10/25 15:59
@doc: 
"""
import numpy as np
import loader
import utils


class pam(object):
    def __init__(self, k, file, max_iter=50, precision=5):
        self.k = k
        self.file = file
        self.max_iter = max_iter
        self.precision = precision

    def calc_pam(self):
        raw_data = loader.file_load(self.file)
        centroid = utils.init_center(raw_data, self.k)