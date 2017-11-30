#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vo4f
@project: ClusterHomework
@file: svm
@time: 2017/11/29
@doc: 
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC

import loader


class SVM(object):
    def __init__(self):
        self.X = None
        self.Y = None

    def fit(self, fpath, name, num):
        face_data, face_label = loader.face_load(fpath, name, num)
        pca = PCA(n_components=16, svd_solver='auto', whiten=True)
        pca.fit(face_data)
        pca_data = pca.transform(face_data)
        self.X = np.array(pca_data)
        self.Y = np.array(face_label)
        plt.figure(1)
        kernel_list = ['rbf', 'poly', 'sigmcid']
        for knl in kernel_list:
            label_x = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
            label_y = []
            for g in label_x:
                label_y.append(self.svm_precision(knl, g))
                plt.plot(label_x, label_y, label=knl)
        plt.xlabel('gamma')
        plt.ylabel('precision')
        plt.legend()
        plt.show()

    def svm_precision(self, knl, g):
        clf = self.svm_create(knl, g)
        kf = KFold(n_splits=10, shuffle=True)
        precision_average = 0.0
        for train, test in kf.split(self.X):
            clf = clf.fit(self.X[train], self.Y[train])
            test_pred = clf.predict(self.X[test])
            count = 0
            for i in range(0, len(self.Y[test])):
                if self.Y[test][i] == test_pred[i]:
                    count += 1
            precision_average = precision_average + float(count) / len(self.Y[test])
        precision_average = precision_average / 10
        print("precison: ", str(precision_average))
        return precision_average

    def svm_create(self, knl, g):
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}
        clf = GridSearchCV(SVC(kernel=knl, class_weight='balanced', gamma=g), param_grid)
        return clf
