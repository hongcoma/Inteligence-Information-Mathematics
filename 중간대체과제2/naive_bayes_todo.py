# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:52:28 2016

@author: JeeHang Lee
@date: 20160926
@description: This is an example code showing how to use Naive Bayes 
        implemented in scikit-learn.  
"""

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

import pandas as pd
import numpy as np

def replace(df):
    df = df.replace(['paid', 'current', 'arrears'], [2, 1, 0])
    df = df.replace(['none', 'guarantor', 'coapplicant'], [0, 1, 1])
    df = df.replace(['coapplicant'], [1])
    df = df.replace(['rent', 'own'], [0, 1])
    df = df.replace(['False', 'True'], [0, 1])
    df = df.replace(['none'], [float('NaN')])
    df = df.replace(['free'], [-1])
    return df
    
df = pd.read_csv('./fraud_data.csv')
res = replace(df)

#print(res)
history = np.array(res.History)
coapplicant = np.array(res.CoApplicant)
accommodation = np.array(res.Accommodation)

X = np.array([[history[i],coapplicant[i],accommodation[i]] for i in range(20)])
Y = np.array(res.Fraud)

model = GaussianNB()
model.fit(X, Y)
predicted = model.predict([[2,0,0],[2,1,0],[0,1,0],[0,1,1],[0,1,1]])
pred_prob = model.predict_proba([[2,0,0],[2,1,0],[0,1,0],[0,1,1],[0,1,1]])
print(predicted)
print(pred_prob)



'''
X = np.array([[history[i],coapplicant[i],accommodation[i]] for i in range(20)])

X = np.array([[history[0],coapplicant[0],accommodation[0]],
           [history[1],coapplicant[1],accommodation[1]],
           [history[2],coapplicant[2],accommodation[2]],
           [history[3],coapplicant[3],accommodation[3]],
           [history[4],coapplicant[4],accommodation[4]],
           [history[5],coapplicant[5],accommodation[5]],
           [history[6],coapplicant[6],accommodation[6]],
           [history[7],coapplicant[7],accommodation[7]],
           [history[8],coapplicant[8],accommodation[8]],
           [history[9],coapplicant[9],accommodation[9]],
           [history[10],coapplicant[10],accommodation[10]],
           [history[11],coapplicant[11],accommodation[11]],
           [history[12],coapplicant[12],accommodation[12]],
           [history[13],coapplicant[13],accommodation[13]],
           [history[14],coapplicant[14],accommodation[14]],
           [history[15],coapplicant[15],accommodation[15]],
           [history[16],coapplicant[16],accommodation[16]],
           [history[17],coapplicant[17],accommodation[17]],
           [history[18],coapplicant[18],accommodation[18]],
           [history[19],coapplicant[19],accommodation[19]]])

X = np.array([[1,0,1],[2,0,1],[2,0,1],[2,1,0],[0,0,1],
              [0,0,1],[1,0,1],[0,0,1],[1,0,0],[0,0,1],
              [1,1,1],[1,0,1],[1,0,0],[2,0,1],[0,0,1],
              [1,0,1],[0,1,0],[0,0,-1],[0,0,1],[2,0,1]])

Y = np.array(res.Fraud)

Y = np.array([[1],[0],[0],[1],[0],
              [1],[0],[0],[0],[1],
              [0],[1],[1],[0],[0],
              [0],[0],[0],[0],[0]])

Y = np.array([res.Fraud[0],res.Fraud[1],res.Fraud[2],res.Fraud[3],res.Fraud[4],
              res.Fraud[5],res.Fraud[6],res.Fraud[7],res.Fraud[8],res.Fraud[9],
              res.Fraud[10],res.Fraud[11],res.Fraud[12],res.Fraud[13],res.Fraud[14],
              res.Fraud[15],res.Fraud[16],res.Fraud[17],res.Fraud[18],res.Fraud[19]])
'''