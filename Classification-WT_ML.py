# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:13:04 2018

@author: Winham

===========基于小波系数特征和支持向量机SVM的ECG分类算法=============

*需要第三方工具包numpy,h5py,scikit-learn
*涉及函数的使用方法可自行查看工具包文档，baidu即可得
*ECG算法入门系列博客：https://blog.csdn.net/qq_15746879
*开源github：https://github.com/Aiwiscal

*本代码所需要的特征和标签文件来自matlab提取
*详情：https://blog.csdn.net/qq_15746879/article/details/80340910
================================================================
"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#设定路径及文件名并载入，这里的特征来自Matlab中提取的小波系数
#https://blog.csdn.net/qq_15746879/article/details/80340910

print("Loading Features and Labels...")
Path='F:/Python files/ECGPrimer/' #自定义路径要正确
FeatureFile='Feature_WT_25.mat'
LabelFile='Label.mat'

tic=time.time()
Feature=load_mat(Path+FeatureFile,'Feature')
Feature=Feature.T
Label=load_mat(Path+LabelFile,'Label')
Label=Label.squeeze()
toc=time.time()

print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

#划分数据集并归一化
print("Data normalizing...")
tic=time.time()
train_x,test_x,train_y,test_y=train_test_split(Feature,Label, test_size=0.5, random_state=42)
min_max_scaler=preprocessing.MinMaxScaler() 
train_x=min_max_scaler.fit_transform(train_x)
test_x= min_max_scaler.transform(test_x)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

#模型训练及预测
print("SVM training and testing...")
tic=time.time()
SVM=SVC(kernel='rbf',C=2,gamma=1)
SVM.fit(train_x,train_y)
y_pred=SVM.predict(test_x)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

#评估结果
Acc=np.mean(y_pred==test_y)
Conf_Mat=confusion_matrix(test_y,y_pred) #利用专用函数得到混淆矩阵
Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])


print('\nAccuracy=%.2f%%'%(Acc*100))
print('Accuracy_N=%.2f%%'%(Acc_N*100))
print('Accuracy_V=%.2f%%'%(Acc_V*100))
print('Accuracy_R=%.2f%%'%(Acc_R*100))
print('Accuracy_L=%.2f%%'%(Acc_L*100))
print('\nConfusion Matrix:\n')
print(Conf_Mat)

print("======================================")