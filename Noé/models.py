import numpy as np
import matplotlib.pyplot as plt
import random

from implementations import *


def baseline_bphig4(x): #classifies only based on bphig4 column
    n,_=x.shape
    base = np.ones(n)
    for i in range(n):
        if (x[i,34] != 1.): base[i]=-1
    return base

def mse_submission(y_predict,y_true): #to calculate mse of a prediction
    n=y_predict.shape[0]
    mse=np.sum((y_predict-y_true)**2) /(2*n)
    return mse

def f1_submission(y_predict,y_true): #to calculate f1-score of a prediction
    n=y_predict.shape[0]
    tp,fn,fp=0,0,0
    for i in range(n):
        if y_predict[i]==1:
            if y_true[i]==1: tp+=1
            else : fp+=1
        else : 
            if y_true[i]==1: fn+=1
    return tp/(tp+(fp+fn)/2)

def pred_0to1(y): #converts 0/1 to -1/1
    n=y.shape[0]
    out=np.ones(n)
    for i in range(n):
        if y[i]==0: out[i]=-1
    return out

def pred_1to0(y): #converts -1/1 to 0/1
    n=y.shape[0]
    out=np.ones(n)
    for i in range(n):
        if y[i]==-1: out[i]=0
    return out

def class_0(y): #for predictions made in 0/1, classifies over/under 0.5
    n=y.shape[0]
    out=np.ones(n)
    for i in range(n):
        if y[i]<0.5: out[i]=0
    return out

def class_1(y): #for predictions made in -1/1, classifies over/under 0
    n=y.shape[0]
    out=np.ones(n)
    for i in range(n):
        if y[i]<0: out[i]=-1
    return out

def nan_to_zero(x): #as a starter, treat all nans as 0
    return np.nan_to_num(x)

def standardize(x): #to standardize the cleaned dataset
    (n,p)=x.shape
    means=np.zeros((n,p))
    stds=np.zeros((n,p))
    for i in range(p):
        moy=np.mean(x[:,i])
        for j in range(n): means[j,i]=moy
        std=np.std(x[:,i])
        for j in range(n): stds[j,i]=std
    std_data=(x-means)/stds 
    return std_data

def reg_log_reg_test(y,tx,lambda_ ,initial_w, max_iters, gamma): #trains the model using GD with logistic loss and regularizer, returns the model, the prediction made on the training set, the nll,mse and f1-score of this prediction
    y_tozero=pred_1to0(y)
    x_clean=standardize(nan_to_zero(tx))
    w,nll_train=reg_logistic_regression(y_tozero, x_clean, lambda_ ,initial_w, max_iters, gamma)
    y_pred=x_clean.dot(w)
    out_train=pred_0to1(class_0(y_pred))
    mse_train,f1_train=mse_submission(out_train,y),f1_submission(out_train,y)
    return w,out_train,nll_train,mse_train,f1_train
    