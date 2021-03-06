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
        self.kernel_list = ['rbf', 'poly', 'sigmoid']
        self.pre_list = []

    def fit(self, fpath, name, num):
        face_data, face_label = loader.face_load(fpath, name, num)
        pca = PCA(n_components=16, svd_solver='auto', whiten=True)
        pca.fit(face_data)
        pca_data = pca.transform(face_data)
        self.X = np.array(pca_data)
        self.Y = np.array(face_label)
        for knl in self.kernel_list:
            print('In', knl)
            tmp_pre = self.svm_precision(knl)
            self.pre_list.append(tmp_pre)

    def svm_precision(self, knl):
        clf = self.svm_create(knl)
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
        print("precision: ", str(precision_average))
        print("best parameter: ", clf.best_params_)
        return precision_average

    def svm_create(self, knl):
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        clf = GridSearchCV(SVC(kernel=knl, class_weight='balanced'), param_grid)
        return clf
