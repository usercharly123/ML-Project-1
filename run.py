import numpy as np
import matplotlib.pyplot as plt
import random

from implementation import *
from helpers import load_csv_data, create_csv_submission

# load dataset
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("C:/Users/Tistou/Desktop/Cours EPFL/MA3/Machine Learning/Projet1/dataset")

ind_5000 = random.sample(range(328135),5000) #for testing on a subgroup of 5000 persons
x_train_4000 = x_train[ind_5000[:4000]]
y_train_4000 = y_train[ind_5000[:4000]]
x_test_1000 = x_train[ind_5000[4000:5000]]
y_test_1000 = y_train[ind_5000[4000:5000]]

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

def nan_to_mean(x,l): #instead of setting nans to 0 we set them to the mean of the column values, for the feature indices in l
    n,_=x.shape
    xx=np.copy(x)
    for j in l : 
        sum=0
        non_nan=0
        nan_ind=[]
        for i in range(n): #go through the column once to note where nans are and take the mean of non_nan values
            if np.isnan(x[i,j]):nan_ind.append(i)
            else : 
                sum+=x[i,j]
                non_nan+=1
        mean_j=sum/non_nan
        for i in nan_ind : #change the nans to the mean value
            xx[i,j]=mean_j
    return xx

def build_poly(x,degree):
    ''' Each feature gets applied the degree enhancement, so we end up with a dataset of size n * d.degree
    This way we might be able to capture more complexity in the way features are coded (since linear increase of their value does not always mean linear increas of the output...)
    '''
    n,d=x.shape
    res=np.zeros((n,d*degree))
    for i in range(d):
        res[:,i*degree]=x[:,i]
        for j in range(1,degree):
            res[:,i*degree+j]=res[:,i*degree+j-1]*x[:,i]
    return res

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

y_test_pred_base=baseline_bphig4(x_test) # to generate the baseline based on bphigh4 column only
create_csv_submission(test_ids, y_test_pred_base, 'baseline bphig4')

w_reg_log_30to100,_,_,_,_=reg_log_reg_test(y_train,x_train[:,30:100],0.001,np.zeros(70),100,0.5) # to generate the reg_log 30_100 prediction
y_test_pred_log_30to100 = pred_0to1(class_0(standardize(nan_to_zero(x_test[:,30:100])).dot(w_reg_log_30to100)))
create_csv_submission(test_ids, y_test_pred_log_30to100, 'reg_log 30to100')
'''pas mal, f1=0.380'''
    
def reg_log_reg_poly(y,tx,degree,lambda_ ,initial_w, max_iters, gamma): #logistic regression after building polynomial basis
    y_tozero=pred_1to0(y)
    x_clean=standardize(build_poly(nan_to_zero(tx),degree))
    w,nll_train=reg_logistic_regression(y_tozero, x_clean, lambda_ ,initial_w, max_iters, gamma)
    y_pred=x_clean.dot(w)
    out_train=pred_0to1(class_0(y_pred))
    mse_train,f1_train=mse_submission(out_train,y),f1_submission(out_train,y)
    return w,out_train,nll_train,mse_train,f1_train

w_reg_log,_,_,_,_=reg_log_reg_poly(y_train[:250000],x_train[:250000,30:100],5,0.001,np.zeros(350),500,0.5)
mse_submission(pred_0to1(class_0(standardize(build_poly(nan_to_zero(x_train[250000:,30:100]),5)).dot(w_reg_log))),y_train[250000:]),f1_submission(pred_0to1(class_0(standardize(build_poly(nan_to_zero(x_train[250000:,30:100]),5)).dot(w_reg_log))),y_train[250000:])

w_reg_log_poly_30to100,_,_,_,_=reg_log_reg_poly(y_train,x_train[:,30:100],5,0.001,np.zeros(350),500,0.5)
y_test_pred_log_poly_30to100 = pred_0to1(class_0(standardize(build_poly(nan_to_zero(x_test[:,30:100]),5)).dot(w_reg_log_poly_30to100)))
create_csv_submission(test_ids, y_test_pred_log_poly_30to100, 'reg_log_poly 30to100')
''' f1=0.392 '''
