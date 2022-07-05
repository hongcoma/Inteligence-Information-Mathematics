# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:51:46 2021
@author: Administrator
"""

'''
11월20일 03시48분
1. Testset에서 classification후에 Dataset으로 추가하는 더 좋은 방법 생각하기 (지금방법 너무 구림)
2. Voronoi tessellation 만약 각각 출력한다면 그 방법 생각하기 -> 아마 main에서 append 직전에 출력?
3. 보고서 작성시 각각의 문제를 section화 하는데 코드가 총 5개로 나눠져야하는지? -> 1,2번 메모 해결 -> ★★google doc 질문하기 ★★


11월30일 21시56분
1. voronoi tessellation은 총 4개 출력 -> 이때 각각의 KNN 시행결과의 차이점을 분석, 기술
2. ★★★scater plot에서 각각의 class마다 marker가 다르게 표시되는 방법 찾아서 구현 ★★★ -> 주석형식으로 출력하는것으로 대체 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def Calculate_Distance_Eucildean(row1, row2): #row1은 전체 data set, row2는 test할 data set을 받아서 거리측정하는 def
    DistanceOfEuclidean = [0]*len(row1)
    for i in range(len(row1)):
        DistanceOfEuclidean[i]=((row1[i][0]-row2[0][0])**2 + (row1[i][1]-row2[0][1])**2)**0.5
    return DistanceOfEuclidean

def Calculate_Distance_Manhattan(row1, row2): #row1은 전체 data set, row2는 test할 data set을 받아서 거리측정하는 def
    DistanceOfManhattan = [0]*len(row1)
    for i in range(len(row1)):
        DistanceOfManhattan[i]= abs(row1[i][0] - row2[0][0])+abs(row1[i][1]- row2[0][1])
    return DistanceOfManhattan

def K_Nearest_Neighbor(data, test, k, D): #data는 전체 DataSet, test는 TestSet, D는 어떤 distance측정방법을 사용할 것인지?
    if(D=='E'): #Eucilidean 방법 사용시의 if
        distance_set=Calculate_Distance_Eucildean(data,test) #distance 측정을 해서 distance_set에 save
        NewDataSet=[] #distance들을 저장할 배열 선언
        for i in range(len(data)):
            NewDataSet.append([0 for j in range(2)])
        for i in range(len(data)): #test와의 distance와 그때의 result값 (Draft값)을 저장 
            NewDataSet[i][0]=distance_set[i]
            NewDataSet[i][1]=data[i][2]

        NewDataSet.sort() #distance가 짧은 순으로 배열을 정렬 
        print(k,"- Nearest Neighbor distance And Result (Use of Eucilidean distance)")
        for i in range(k):
            print(NewDataSet[i]) #distance가 짧은 순으로 정렬된 배열을 K개 만큼 출력 
        return NewDataSet
    
    elif(D=="M"):#Manhattan 방법 사용시의 if
        distance_set=Calculate_Distance_Manhattan(data,test) #distance 측정을 해서 distance_set에 save
        NewDataSet=[] #distance들을 저장할 배열 선언
        for i in range(len(data)):
            NewDataSet.append([0 for j in range(2)])
        for i in range(len(data)): #test와의 distance와 그때의 result값 (Draft값)을 저장 
            NewDataSet[i][0]=distance_set[i]
            NewDataSet[i][1]=data[i][2]

        NewDataSet.sort() #distance가 짧은 순으로 배열을 정렬 
        print(k,"- Nearest Neighbor distance And Result (Use of Manhattan distance)")
        for i in range(k):
            print(NewDataSet[i]) #distance가 짧은 순으로 정렬된 배열을 K개 만큼 출력 
        return NewDataSet

def Show_Result(distance_set, k): #실제 결과를 보여주는 def
    count=0 #classification 결과가 1일때 저장할 변수  
    result=2 #최종 결과 (Draft)를 0또는 1로 저장하기 위해 2로 초기화 
    for i in range(k):
        if(distance_set[i][1]==1):
            count+=1
    if(count>int(k/2)): #전체 count 수 중에서 과반수 이상일 경우(K개로 추출했을때 K의 과반수)
        #print("This requirement's classification is Yes(True)\n")
        result=1 #결과값 1저장 
        return result
    else:
        #print("This requirement's classification is No(Flase)\n")
        result=0 #결과값 0저장 
        return result
    
def Print_Voronoi(DataSet,K,distance):
    D=np.array(DataSet)
    x1=np.array(D[:,0])
    x2=np.array(D[:,1])
    y=np.array(D[:,2])
    points=np.delete(DataSet,2,axis=1)
    vor = Voronoi(points)   
    voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_alpha=0.6, point_size=2)
    if(K==3 and distance =="E"):
        qtitle = "3NN with Euclidean"
    elif(K==3 and distance == "M"):
        qtitle = "3NN with Manhattan"
    elif(K==5 and distance == "E"):
        qtitle = "5NN with Euclidean"
    elif(K==5 and distance == "M"):
        qtitle = "5NN with Manhattan"
    plt.title(qtitle)
    for i in range(len(y)):
        if y[i] == 1:
            plt.plot(x1[i],x2[i],'ob')
        elif y[i] == 0:
            plt.plot(x1[i],x2[i],'xr')
    plt.show()

def Show_KNN(k,D):
    DataSet=[  #초기 DataSet
    [2.50, 6.00, 0],[3.75, 8.00, 0],[2.25, 5.50, 0],[3.25, 8.25, 0],
    [2.75, 7.50, 0],[4.50, 5.00, 0],[3.50, 5.25, 0],[3.00, 3.25, 0],
    [4.00, 4.00, 0],[4.25, 3.75, 0],[2.00, 2.00, 0],[5.00, 2.50, 0],
    [8.25, 8.50, 0],[5.75, 8.75, 1],[4.75, 6.25, 1],[5.50, 6.75, 1],
    [5.25, 9.50, 1],[7.00, 4.25, 1],[7.50, 8.0, 1],[7.25, 5.75, 1]
    ]

    TestSet=[[6.75, 3],[5.34, 6.0],[4.67,8.4],[7.0,7.0],[7.8,5.4]] #Test할 TestSet들 (classification 후에 DataSet에 추가될 예정)
    for i in range(5):
        Q=K_Nearest_Neighbor(DataSet, [TestSet[i]],k,D)
        if(Show_Result(Q, k)==1):
            if(i==0): #i번째 순서에 맞게 DataSet에 추가 
                DataSet.append([6.75, 3,1])
            elif(i==1):
                DataSet.append([5.34, 6.0,1])
            elif(i==2):
                DataSet.append([4.67, 8.4,1])
            elif(i==3):
                DataSet.append([7.0, 7.0,1])
            elif(i==4):
                DataSet.append([7.8, 5.4,1])
            print(TestSet[i],"'s classification is Yes(True) \n")
        elif(Show_Result(Q, k)==0):
            if(i==0): #i번째 순서에 맞게 DataSet에 추가 
                DataSet.append([6.75, 3,0])
            elif(i==1):
                DataSet.append([5.34, 6.0,0])
            elif(i==2):
                DataSet.append([4.67, 8.4,0])
            elif(i==3):
                DataSet.append([7.0, 7.0,0])
            elif(i==4):
                DataSet.append([7.8, 5.4,0])
            print(TestSet[i],"'s classification is No(False) \n")
    #print(k,"- NN(with",D,") classfications's DataSet is ",DataSet)
    Print_Voronoi(DataSet,k,D)

Show_KNN(3,"E")
Show_KNN(3,"M")
Show_KNN(5,"E")
Show_KNN(5,"M")


    
    
    