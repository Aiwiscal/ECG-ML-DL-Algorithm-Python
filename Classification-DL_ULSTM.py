# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:53:14 2018

@author: Winham
"""
"""
===================基于单向LSTM(ULSTM)的ECG分类算法========================

*需要第三方工具包numpy,h5py,scikit-learn
*基于深度学习框架TensorFlow
*涉及函数的使用方法可自行查看工具包文档，baidu即可得
*ECG算法入门系列博客：https://blog.csdn.net/qq_15746879
*开源github：https://github.com/Aiwiscal

*本代码所需要的数据和标签文件来自matlab提取
==================================================================
"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sess=tf.InteractiveSession()

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#使用TensorFlow组件完成ULSTM网络的搭建
def ULSTM(x,n_input,n_hidden,n_steps,n_classes):
   
    x=tf.transpose(x,[1,0,2])    #整理数据，使之符合ULSTM接口要求
    x=tf.reshape(x,[-1,n_input])
    x=tf.split(x,n_steps)
    
    #以下两句调用TF函数，生成一个单隐层的ULSTM
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    outputs,_=tf.contrib.rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    
    #以下部分将ULSTM每一步的输出拼接，形成特征向量
    for i in range(n_steps):
        if i==0:
            fv=outputs[0]
        else:
            fv=tf.concat([fv,outputs[i]],1)
    fvp=tf.reshape(fv,[-1,1,n_steps*n_hidden,1])
    shp=fvp.get_shape()
    flatten_shape=shp[1].value*shp[2].value*shp[3].value
    
    fvp2=tf.reshape(fvp,[-1,flatten_shape])
    
    #构建最后的全连接层
    weights=tf.Variable(tf.random_normal([flatten_shape,n_classes]))
    biases=tf.Variable(tf.random_normal([n_classes]))
            
    return tf.matmul(fvp2,weights)+biases

#随机获取一个batch大小的数据，用于训练
def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

#设定路径及文件名并载入，这里的心拍在Matlab下截取完成
#详情：https://blog.csdn.net/qq_15746879/article/details/80340671
Path='F:/Python files/ECGPrimer/' #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("ULSTM setup and initialize...")

n_input=1
n_hidden=1
n_steps=250
n_classes=4
tic=time.time()
x=tf.placeholder(tf.float32, [None, 250]) #定义placeholder数据入口
x_=tf.reshape(x,[-1,250,1])
y_=tf.placeholder(tf.float32,[None,4])

logits=ULSTM(x_,n_input,n_hidden,n_steps,n_classes)

learning_rate=0.001
batch_size=16
maxiters=10000

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
#这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
tf.global_variables_initializer().run()
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("ULSTM training and testing...")
tic=time.time()
for i in range(maxiters):
    batch_x,batch_y=get_batch(train_x,train_y,batch_size)
    train_step.run(feed_dict={x:batch_x,y_:batch_y})
    if i%500==0:
        loss=cost.eval(feed_dict={x:train_x,y_:train_y})
        print("Iteration %d/%d:loss %f"%(i,maxiters,loss))

y_pred=logits.eval(feed_dict={x:test_x,y_:test_y})
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(test_y,axis=1)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

Acc=np.mean(y_pred==y_true)
Conf_Mat=confusion_matrix(y_true,y_pred) #利用专用函数得到混淆矩阵
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