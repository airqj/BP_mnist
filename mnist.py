
# coding: utf-8

# In[167]:


import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid


# In[259]:


# 输入层神经元为 784
# 隐藏层神经元为 20
# 输出层神经元为 10
input_size = 784
hidden_size = 300
output_size = 10
epochs = 20


# In[262]:


Wh  = np.zeros((input_size,hidden_size))
Wo  = np.zeros((hidden_size,output_size))
#Wh = np.random.randn(input_size,hidden_size)
#Wo = np.random.randn(hidden_size,output_size)
#输出层偏置向量
Wob = np.random.randn(output_size)
#Wob = np.zeros(output_size)
#隐藏层偏置向量
Whb = np.random.randn(hidden_size)
#Whb = np.zeros(hidden_size)
learning_rate = 0.01


# In[266]:


pkl_file = open("/home/qinjianbo/Downloads/mnist.pkl","rb")
train_set,valid_set,test_set = pickle.load(pkl_file,encoding="iso-8859-1")
train_X = train_set[0]
train_Y = train_set[1]
valid_X = valid_set[0]
valid_Y = valid_set[1]

train_X = np.concatenate((train_X,valid_X))
train_Y = np.concatenate((train_Y,valid_Y))

test_X = test_set[0]
test_Y = test_set[1]


# In[238]:


#输出层节点误差项计算公式
#P :预测值
#T :真实值
def Output_diff(P,T):
    diff = P * (1 - P) * (T - P)
    return diff

def get_loss(P,T):
    return np.square(P - T).sum() / P.shape[0]


# In[270]:


# 输入为x
for epoch in range(epochs):
    loss = 0.0
    for k in range(train_Y.shape[0]):
        x = train_X[k]
        Yreal = np.array([0.1] * 10)
        Yreal[train_Y[k]] = 0.9

        #前向传播
        H = sigmoid(np.matmul(Wh.T,x) + Whb) #隐藏层的值
        Ypredict = sigmoid(np.matmul(Wo.T,H) + Wob) #输出层的值

        #输出层偏导矩阵
        Wo_update_mat = np.zeros(Wo.shape)
        #隐藏层偏置更新向量
        Whb_update_mat = np.zeros(Whb.shape)
        #输出层偏置更新向量
        Wob_update_mat = np.zeros(Wob.shape)

        Ydiff = Output_diff(Ypredict,Yreal)
        for index in range(Ypredict.shape[0]):
            #Yindex = Ypredict[index] * (1 - Ypredict[index]) * (Yreal[index] - Ypredict[index])
            #Yindex = Output_diff(Ypredict[index],Yreal[index])
            Yindex = Ydiff[index]
            update_entry =  learning_rate * Yindex * H
            update_entry = update_entry.reshape(hidden_size)
            Wo_update_mat[:,index] = update_entry
            Wob_update_mat[index] = learning_rate * Yindex

        #Ydiff 是输出层误差项
        #Hdiff 是隐藏层误差项
        Hdiff = np.array([0] * H.shape[0] , dtype=float)
        for j in range(len(H)):
            asum = 0
            for i in range(len(Ydiff)):
                asum += Wo.T[i][j] * Ydiff[i]
            Hdiff[j] = H[j] * (1 - H[j]) * asum

        Wh_update_mat = np.zeros(Wh.shape)
        for index in range(H.shape[0]):
            Hindex = Hdiff[index]
            update_entry = learning_rate * Hindex * x
            update_entry = update_entry
            Wh_update_mat[:,index] = update_entry
            Whb_update_mat[index] = learning_rate * Hindex


        Wo += Wo_update_mat
        Wh += Wh_update_mat

        Whb +=Whb_update_mat
        Wob +=Wob_update_mat

        loss += get_loss(Ypredict,Yreal)
        if(k % 3000 == 0):
            print("[%s|%s] in %s,loss is %s"%(epoch,epochs,k,loss/3000))
            loss = 0
    print("finish %s eopch"%(epoch))
    print("calculating accuracy")
    result_predict = np.array([-1] * valid_Y.shape[0])
    for index_test_X in range(test_X.shape[0]):
        H = sigmoid(np.matmul(Wh.T,test_X[index_test_X]) + Whb) #隐藏层的值
        Ypredict = sigmoid(np.matmul(Wo.T,H) + Wob) #输出层的值
        result_predict[index_test_X] = np.argmax(Ypredict)
    zero_vec = (result_predict - test_Y)
    zero_count = len(zero_vec[zero_vec == 0])
    print("[%s|%s] valid acc is %s"%(epoch+1,epochs,zero_count/test_Y.shape[0]))
