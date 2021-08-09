import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Dense, Dropout, Lambda, Concatenate, AvgPool2D, MaxPool2D, \
    Softmax,Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model
import h5py
import numpy as np
from model import Calculate
import torch
import scipy.io as scio
path='D:/network_graduation/xuezhang/deap_test.mat'
data = h5py.File(path)
input = np.array(data['specf_test'])
input = input.transpose((3, 2, 1, 0))
label = np.array(data['label_test'])
label = label.transpose((1, 0))
label_compute=[]
new_label=[0]*1600
new_input=[]
model=Calculate(32,32,5,5,batchsize=1)
for i in range(input.shape[0]):
    now=[]
    for j in range(input.shape[0]):
        if label[i]==label[j]:
            new_label[40*i+j]=1
        else:
            new_label[40*i+j]=0
        aa=np.expand_dims(input[i,:,:,:],axis=0)
        bb=np.expand_dims(input[j,:,:,:],axis=0)
        new_input.append(np.concatenate((aa,bb),axis=0))
        a=tf.expand_dims(tf.convert_to_tensor(input[i,:,:,:],dtype=np.float),axis=0)
        b=tf.expand_dims(tf.convert_to_tensor(input[j,:,:,:],dtype=np.float),axis=0)
        c=model([a,b]).numpy().item()
        print(tf.reduce_sum(a),tf.reduce_sum(b),c)
        now.append(c)
    label_compute.append(now)
new_label=np.array(new_label,dtype=np.float)
new_input=np.array(new_input,dtype=np.float)
label_compute=np.array(label_compute,dtype=np.float)
print(new_label.shape,new_input.shape)
np.save('neighbor_array.npy',label_compute)
np.save('new_label.npy',new_label)
np.save('new_input.npy',new_input)