#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:44:18 2020

@author: waynelee
"""
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from skrebate import ReliefF


def get_importance(X_train, y_train, name):
    if name == "gini":
       forest = ExtraTreesClassifier(n_estimators=250, criterion="gini",random_state=42)
       forest.fit(X_train, y_train)
       importances = forest.feature_importances_
    elif name == "entropy":
       forest = ExtraTreesClassifier(n_estimators=250, criterion="entropy",random_state=42)
       forest.fit(X_train, y_train)
       importances = forest.feature_importances_
    elif name == "mutual_info_classif":
       importances =  mutual_info_classif(X_train, y_train)
    elif name == "chi2":
       importances =  chi2(X_train, y_train)[0]
    elif name == "f_classif":
       importances =  f_classif(X_train, y_train)[0]
    elif name == 'ReliefF':
        rr = ReliefF()
        rr.fit(X_train, y_train)
        importances = rr.feature_importances_
    return importances   
    
    