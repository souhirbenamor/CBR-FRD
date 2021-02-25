#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:50:29 2020

@author: wayne
"""
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,precision_recall_fscore_support

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV

def measure_others(X_train, y_train,X_test,y_test):
    # Logistic Regression 
    log_reg_params = {"penalty": ['none', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)
    # We automatically get the logistic regression with the best parameters.
    log_reg = grid_log_reg.best_estimator_
    
    knears_params = {"n_neighbors": list(range(1,10,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    
    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
    grid_knears.fit(X_train, y_train)
    # KNears best estimator
    knears_neighbors = grid_knears.best_estimator_
    k = grid_knears.best_estimator_.n_neighbors # k nearest 
    
    # Support Vector Classifier
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    grid_svc = GridSearchCV(SVC(), svc_params)
    grid_svc.fit(X_train, y_train)
    
    # SVC best estimator
    svc = grid_svc.best_estimator_
    
    # DecisionTree Classifier
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
                  "min_samples_leaf": list(range(5,7,1))}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    grid_tree.fit(X_train, y_train)
    
    # tree best estimator
    tree_clf = grid_tree.best_estimator_
    ########################################################################################
    
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_knear = knears_neighbors.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_tree = tree_clf.predict(X_test)
    
    ###############################################################################################    
    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
    ######################################################################################## 
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(objective="binary:logistic")   
    xgb_params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1 
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.6, 0.4)
    }
    
    rand_xgb = RandomizedSearchCV(xgb_model, param_distributions=xgb_params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)
    
    rand_xgb.fit(X_train, y_train)
    
    # KNears best estimator
    xgb = rand_xgb.best_estimator_
    
    y_pred_xgb = xgb.predict(X_test)

    return k, y_pred_log_reg, y_pred_knear, y_pred_svc, y_pred_tree, y_pred_gnb, y_pred_xgb
