#!/usr/bin/env python3
# encoding: utf-8
"""
@author: vo4f
@project: PyCharm
@file: dataproc.py
@time: 2017/10/22 20:25
@doc: Process basic data
"""
import csv

list0 = []
list1 = []
list2 = []
with open('waveform.data', 'r') as f:
    raw = list(csv.reader(f, delimiter=","))
    for i in raw[:500]:
        if i[-1] == '0' and len(list0) < 100:
            list0.append(i)
        elif i[-1] == '1' and len(list1) < 100:
            list1.append(i)
        elif i[-1] == '2' and len(list1) < 100:
            list2.append(i)

list012 = list0 + list1 + list2

with open('waveform012.data', 'w', newline='') as f:
    wf = csv.writer(f)
    wf.writerows(list012)
