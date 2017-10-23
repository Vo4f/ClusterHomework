#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: Loader.py
@time: 2017/10/22 20:29
@doc: File Loader
"""


class Loader:
    def __init__(self, file_name, type):
        self.file_name = file_name
        self.type = type

    def loader(self):
        if self.type == 'text':
            self

    def text_loader(self):
        with open(self.file_name, 'r') as file:
            pass