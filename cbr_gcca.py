"""
Created on Thu Oct 29 20:17:56 2020

@author: waynelee
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_X_y
from sklearn.utils import check_array
import math
import time
import random
import codecs, json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

from timeit import default_timer as timer
from collections import Counter
from numba import jit
from numba import cuda
from numba import *
import pyswarms as ps
import logging
cuda_logger = logging.getLogger('numba.cuda.cudadrv.driver')
cuda_logger.setLevel(logging.ERROR)  # only show error

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from measure_others import measure_others 
from get_importance import get_importance

def dist_computation(w, p, p1, m, r,q):
    f = m.shape[0]
    sim = 0.0
    for i in range(f):
       if q[i] >= r[i]:
           sim += ((((maxmin[i] - abs(q[i] - r[i]))/maxmin[i])**p[i])**2)*w[i]
       else:
           sim += ((((maxmin[i] - abs(q[i] - r[i]))/maxmin[i])**p1[i])**2)*w[i]
    sim = math.sqrt(sim)
    return sim

def get_result(k, dist, y_train, y_val):
    n = y_val.shape[0]
    predict = np.zeros((n,))
    for i in range(n):
        a = y_train[dist[i,:].argsort()[-k:]].tolist()     
        if sum(a) == len(a)/2:
            predict[i] = random.randint(0,1)
        else:   
            predict[i] = max(set(a), key=lambda x: a.count(x))
    return accuracy_score(y_val,predict)# from sklearn.linear_model import Lasso

@cuda.jit
def compute_sim(w, p, p1, m, ref, query, k, dist):
    height = query.shape[0]
    width = ref.shape[0]
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    
    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            dist[y,x] = dist_gpu(w, p, p1, m, ref[x,:], query[y,:])

def test(w,p,p1,m,q,r):
    f = m.shape[0]
    sim = 0.0
    for i in range(f):
        sim += ((((maxmin[i] - abs(q[i] - r[i]))/maxmin[i])**p[i])**2)*w[i]
    sim = math.sqrt(sim)
    
    return sim

def get_result1(k, dist, y_train, y_test):
    n = y_test.shape[0]
    predict = np.zeros((n,))
    for i in range(n):
        a = y_train[dist[i,:].argsort()[-k:]].tolist()
        if sum(a) == len(a)/2:
            predict[i] = random.randint(0,1)
        else:   
            predict[i] = max(set(a), key=lambda x: a.count(x))
    return predict
###############################################################################################
#Importing the data
data = pd.read_csv('german.csv')
print(data.isnull().values.any())
###############################################################################################
rob_scaler = RobustScaler()
for column in data.columns:
    if column != 'credit_risk':
        data[column] = rob_scaler.fit_transform(data[column].values.reshape(-1,1))
###############################################################################################
seed = 42
data_X_train = []
data_y_train = []
data_X_test = []
data_y_test = []
for jjj in range(10):
    print("experiment: ",jjj)
    random_t = jjj
    data = data.sample(frac=1, random_state=random_t)     
    # amount of fraud classes    
    good_data = data.loc[data['credit_risk'] == 1][:Counter(data['credit_risk'])[0]]
    bad_data = data.loc[data['credit_risk'] == 0]  
    normal_distributed_data = pd.concat([good_data, bad_data]) 
    # Shuffle dataframe rows
    new_data = normal_distributed_data.sample(frac=1, random_state=42)
    X = new_data.drop('credit_risk', axis=1)
    y = new_data['credit_risk']
    print(Counter(y))
    #############################################################################################
    #############################################################################################
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    scaler = MinMaxScaler()
    y = np.array(y, dtype='float')
    ###############################################################################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    data_X_train.append(X_train.tolist())
    data_y_train.append(y_train.tolist())
    data_X_test.append(X_test.tolist())
    data_y_test.append(y_test.tolist())
    ###############################################################################################
    k, y_pred_log_reg, y_pred_knear, y_pred_svc, y_pred_tree, y_pred_gnb, y_pred_xgb = measure_others(X_train, y_train,X_test,y_test)
    y_pred_all = [y_test.tolist(),y_pred_log_reg.tolist(), y_pred_knear.tolist(), y_pred_svc.tolist(), y_pred_tree.tolist(), y_pred_gnb.tolist(), y_pred_xgb.tolist()]
    ##############################################################################################
    nf = X.shape[1]
    maxmin = np.array(np.max(X, 0))-np.array(np.min(X, 0))
    k0=k
    ###############################################################################################
    dist_gpu = cuda.jit(device=True)(dist_computation) 
    X_train0 = X_train
    y_train0 = y_train
    all_importances = []
    cost_all = []
    poly_all = []
    name = ['gini','entropy','mutual_info_classif','chi2','f_classif','ReliefF']
    k_n = len(name)
    for iii in range(k_n):
        importances = get_importance(X_train,y_train,name[iii])
        print('current iii: ', name[iii])
        kf = KFold(n_splits = 10,random_state=None, shuffle=False)
        def scale_imp(importances):
            b = 1
            a = 0.1
            imp = (b -a) * (importances - importances.min())/(importances.max() - importances.min()) + a
            return imp
        importances0 = importances
        ###################################################################################################
        importances = scale_imp(importances0)
        all_importances.append(importances)
        #####################################################################################################
        start = timer()
        def get_f(params):
            cbr_predict = []
            gimportance =  importances
            weights = gimportance/sum(gimportance)
            polynomial = params[:nf]
            polynomial1 = params[nf:]
            for train_index, val_index in kf.split(X_train0):
                X_train, X_val = X_train0[train_index], X_train0[val_index]
                y_train, y_val = y_train0[train_index], y_train0[val_index]
                dist = np.zeros((len(X_val), len(X_train)), dtype = np.float)
                blockdim = (32, 8)
                grid_x = math.ceil(X_val.shape[0]/32)+10
                grid_y = math.ceil(X_train.shape[0]/8)+10
                griddim = (grid_x,grid_y)
                d_train = cuda.to_device(X_train)
                d_val = cuda.to_device(X_val)
                d_dist = cuda.to_device(dist)
                compute_sim[griddim, blockdim](weights, polynomial,polynomial1, maxmin, d_train, d_val, k, d_dist) 
                d_dist.to_host()
                cbr_predict.append(get_result(k,dist,y_train,y_val))
            cbr_mean = np.mean(np.array(cbr_predict))
            return(-cbr_mean)
        #pso
        swarm_size = nf
        dim = nf       # Dimension of X
        epsilon = 1.0
        options = {'c1': 1.5, 'c2':1.5, 'w':0.8}
        
        swarm_size = nf*2
        dim = nf*2       # Dimension of X
        constraints_p = (np.ones((nf*2,))/10,
                          np.ones((nf*2,))*10)
    
        def opt_func(X):
            n_particles = X.shape[0]  # number of particles
            dist = [get_f(X[i]) for i in range(n_particles)]
            return np.array(dist)
        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                            dimensions=dim,
                                            options=options,
                                            bounds=constraints_p,init_pos = np.array([np.ones((2*nf,)),]*dim))

        cost, joint_vars = optimizer.optimize(opt_func, iters=500)
        dt = timer() - start
        print("Data compute time %f s" % dt)
        cost_all.append(cost)
        poly_all.append(joint_vars)       
        polynomial = joint_vars
        polynomials = polynomial[:nf]
        polynomials1 = polynomial[nf:] 
        ################################################################################################
        dist = np.zeros((len(X_test), len(X_train)), dtype = np.float)
        sims = np.zeros((len(X_test), k), dtype = np.float)
        blockdim = (32, 8)
        grid_x = math.ceil(X_test.shape[0]/32)+10
        grid_y = math.ceil(X_train.shape[0]/8)+10
        griddim = (grid_x,grid_y)
        d_train = cuda.to_device(X_train)
        d_val = cuda.to_device(X_test)
        d_dist = cuda.to_device(dist)
        d_sims = cuda.to_device(sims)
        compute_sim[griddim, blockdim](importances, polynomials, polynomials1, maxmin, d_train, d_val, k, d_dist) 
        d_dist.to_host()
        y_pred_cbr_e = get_result1(k,dist,y_train,y_test)
        print('E-CBR Classifier:')
        print(classification_report(y_test, y_pred_cbr_e))
        print('E-CBR Classifier:',precision_recall_fscore_support(y_test, y_pred_cbr_e, average=None,labels=[1]))
        ################################################################################################
    importances = np.ones((nf,))
    start = timer()   
    def get_f(params):
        cbr_predict = []
        gimportance =  importances
        weights = gimportance/sum(gimportance)
        polynomial = params[:nf]
        polynomial1 = params[nf:]
        for train_index, val_index in kf.split(X_train0):
            X_train, X_val = X_train0[train_index], X_train0[val_index]
            y_train, y_val = y_train0[train_index], y_train0[val_index]
            dist = np.zeros((len(X_val), len(X_train)), dtype = np.float)
    
            blockdim = (32, 8)
            grid_x = math.ceil(X_val.shape[0]/32)+10
            grid_y = math.ceil(X_train.shape[0]/8)+10
            griddim = (grid_x,grid_y)
            d_train = cuda.to_device(X_train)
            d_val = cuda.to_device(X_val)
            d_dist = cuda.to_device(dist)
            compute_sim[griddim, blockdim](weights, polynomial,polynomial1, maxmin, d_train, d_val, k, d_dist) 
            d_dist.to_host()
            cbr_predict.append(get_result(k,dist,y_train,y_val))
        cbr_mean = np.mean(np.array(cbr_predict))
        return(-cbr_mean)  
    #pso
    swarm_size = nf
    dim = nf       # Dimension of X
    epsilon = 1.0
    options = {'c1': 1.5, 'c2':1.5, 'w':0.8}
    
    swarm_size = nf*2
    dim = nf*2       # Dimension of X
    constraints_p = (np.ones((nf*2,))/10,
                      np.ones((nf*2,))*10)
    
    
    def opt_func(X):
        n_particles = X.shape[0]  # number of particles
        dist = [get_f(X[i]) for i in range(n_particles)]
        return np.array(dist)
    
    
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        dimensions=dim,
                                        options=options,
                                        bounds=constraints_p,init_pos = np.array([np.ones((2*nf,)),]*dim))

    cost, joint_vars = optimizer.optimize(opt_func, iters=500)
    dt = timer() - start
    print("Data compute time %f s" % dt) 
    polynomials = joint_vars[:nf]
    polynomials1 = joint_vars[nf:] 
    ################################################################################################
    dist = np.zeros((len(X_test), len(X_train)), dtype = np.float)
    sims = np.zeros((len(X_test), k), dtype = np.float)
    blockdim = (32, 8)
    grid_x = math.ceil(X_test.shape[0]/32)+10
    grid_y = math.ceil(X_train.shape[0]/8)+10
    griddim = (grid_x,grid_y)
    d_train = cuda.to_device(X_train)
    d_val = cuda.to_device(X_test)
    d_dist = cuda.to_device(dist)
    d_sims = cuda.to_device(sims)
    compute_sim[griddim, blockdim](importances, polynomials, polynomials1, maxmin, d_train, d_val, k, d_dist) 
    d_dist.to_host() 
    y_pred_cbr_eqe = get_result1(k,dist,y_train,y_test)
    print('EQE-CBR Classifier:')
    print(classification_report(y_test, y_pred_cbr_eqe))
    print('EQE-CBR Classifier:',precision_recall_fscore_support(y_test, y_pred_cbr_eqe, average=None,labels=[1]))
    y_pred_all.append(y_pred_cbr_eqe.tolist())
    ################################################################################################   
    m = min(cost_all)
    min_i = [i for i, j in enumerate(cost_all) if j == m]
    importances = all_importances[min_i[0]]
    polynomial = poly_all[min_i[0]]
    polynomials = polynomial[:nf]
    polynomials1 = polynomial[nf:] 
    ################################################################################################
    dist = np.zeros((len(X_test), len(X_train)), dtype = np.float)
    sims = np.zeros((len(X_test), k), dtype = np.float)
    blockdim = (32, 8)
    grid_x = math.ceil(X_test.shape[0]/32)+10
    grid_y = math.ceil(X_train.shape[0]/8)+10
    griddim = (grid_x,grid_y)
    d_train = cuda.to_device(X_train)
    d_val = cuda.to_device(X_test)
    d_dist = cuda.to_device(dist)
    d_sims = cuda.to_device(sims)
    compute_sim[griddim, blockdim](importances, polynomials, polynomials1, maxmin, d_train, d_val, k, d_dist) 
    d_dist.to_host() 
    y_pred_cbr_e = get_result1(k,dist,y_train,y_test)
    print('E-CBR Classifier:')
    print(classification_report(y_test, y_pred_cbr_e))
    print('E-CBR Classifier:',precision_recall_fscore_support(y_test, y_pred_cbr_e, average=None,labels=[1]))
    y_pred_all.append(y_pred_cbr_e.tolist())
    ################################################################################################
    dist = np.zeros((len(X_test), len(X_train0)), dtype = np.float)
    sims = np.zeros((len(X_test), k), dtype = np.float)
    blockdim = (32, 8)
    grid_x = math.ceil(X_test.shape[0]/32)+10
    grid_y = math.ceil(X_train0.shape[0]/8)+10
    griddim = (grid_x,grid_y)
    d_train = cuda.to_device(X_train0)
    d_val = cuda.to_device(X_test)
    d_dist = cuda.to_device(dist)
    d_sims = cuda.to_device(sims)
    compute_sim[griddim, blockdim](np.ones((nf,)), np.ones((nf,)), np.ones((nf,)), maxmin, d_train, d_val, k, d_dist) 
    d_dist.to_host()
    y_pred_cbr_eq = get_result1(k,dist,y_train0,y_test)
    print('Equal CBR Classifier:')
    print(classification_report(y_test, y_pred_cbr_eq))
    y_pred_all.append(y_pred_cbr_eq.tolist())
    ##################################################################################################  
    print('log_reg: ',precision_recall_fscore_support(y_test, y_pred_log_reg, average=None))
    print('knn:     ',precision_recall_fscore_support(y_test, y_pred_knear, average=None))
    print('svc:     ',precision_recall_fscore_support(y_test, y_pred_svc, average=None))
    print('d-tree:  ',precision_recall_fscore_support(y_test, y_pred_tree, average=None))
    print('nb:      ',precision_recall_fscore_support(y_test, y_pred_gnb, average=None))
    print('cbr_eq:  ',precision_recall_fscore_support(y_test, y_pred_cbr_eq, average=None))
    print('cbr_e:   ',precision_recall_fscore_support(y_test, y_pred_cbr_e, average=None))
    ###################################################################################################
    print('log_reg: ',roc_auc_score(y_test, y_pred_log_reg))
    print('knn:     ',roc_auc_score(y_test, y_pred_knear))
    print('svc:     ',roc_auc_score(y_test, y_pred_svc))
    print('d-tree:  ',roc_auc_score(y_test, y_pred_tree))
    print('nb:      ',roc_auc_score(y_test, y_pred_gnb))
    print('cbr_eqe:   ',roc_auc_score(y_test, y_pred_cbr_eqe))
    print('cbr_eq:  ',roc_auc_score(y_test, y_pred_cbr_eq))   
    print('cbr_e:   ',roc_auc_score(y_test, y_pred_cbr_e))




    




