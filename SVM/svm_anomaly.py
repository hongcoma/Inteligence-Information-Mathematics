# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 00:32:06 2016

@author: Naver
"""

from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

def make_svmdata_from_file(fname):
    dlist = []
    fp = open(fname,"r")
    for line in fp:
        tmp = []
        line = line.strip("\r\n")
        psd = line.split(",")
        for td in psd:
            if len(tmp) >= 4: break ## valid dimension
            tmp.append(float(td))
        dlist.append(tmp)
    fp.close()
    return dlist    
    
#1. read svm input ( train, test)
train = make_svmdata_from_file("p.train")
test = make_svmdata_from_file("p.train")

#2. svm test
X_train = np.array(train)
X_test = np.array(test)

anomalymodel = OneClassSVM(nu=0.12, kernel='rbf', gamma=0.1)
anomalymodel.fit(X_train)
p = anomalymodel.predict(X_test)

pos=0
for pd in p:
    pos += 1
    if pd == -1:
        print ("anomaly detection", pos - 1)

#3. ploting
plt.subplot(2,1,1)
plt.plot(X_train)
plt.subplot(2,1,2)
plt.plot(X_test)
plt.show()
