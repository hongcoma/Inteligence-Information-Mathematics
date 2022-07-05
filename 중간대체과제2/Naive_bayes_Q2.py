# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 05:40:27 2021

@author: Administrator
"""

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def replace(df):
    df = df.replace(['Male', 'Female'], [1, 0])
    return df

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    
df = pd.read_csv('./Social_Network_Ads.csv')
res = replace(df)

print(res)
#print(res)
gender = np.array(res.Gender)
age = np.array(res.Age)
estimatedsalary = np.array(res.EstimatedSalary)
purchased=np.array(res.Purchased)

X = np.array([[gender[i],age[i],estimatedsalary[i]] for i in range(400)])
Y = np.array(purchased)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=1)
#X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.2,random_state=2)

kfold = KFold(n_splits=10, shuffle=True)
dt = DecisionTreeClassifier()
score = cross_val_score(dt, X, Y, cv=kfold, scoring="accuracy")

model = GaussianNB()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
#pred_prob = model.predict_proba(X_test)
#print(predicted)
#print(pred_prob)

#precision : TP/(TP+FP)
#recall : TP/(TP+FN)
#sensitivity : TP/(TP+FN)
#specificity : TN/(FP+TN)
TN, FP, FN, TP = confusion_matrix(Y_test, predicted).ravel()
print('Hold-Out accuracy_score :',accuracy_score(Y_test, predicted))
print('Hold-Out precision_score :',TP/(TP+FP))
print('Hold-Out recall_score :',TP/(TP+FN))
print('Hold-Out sensitivity_score :',TP/(TP+FN))
print('Hold-Out specificity_score :',TN/(FP+TN))
FPR, TPR, thresholds = roc_curve(Y_test, predicted, pos_label=1)
print('Hold-Out auc score :',auc(FPR, TPR))
print(' 10-fold ave:',score.mean())
print(' 10-fold var:',score.var())
print(FPR, TPR, thresholds)
plot_roc_curve(FPR, TPR)

