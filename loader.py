#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: loader.py
@time: 2017/10/22 20:29
@doc: File Loader
"""
import numpy as np


class Loader(object):
    def __init__(self, file):
        self.file = file

    def load(self):
        return np.load(self.file)
