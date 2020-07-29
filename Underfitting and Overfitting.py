#!/usr/bin/env python
# coding: utf-8

# In[2]:


# impoting the Libraraies 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing the data 

# In[4]:


data = pd.read_csv('data_cleaned.csv')


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# # Segregating the dependent and the independent variable 

# In[8]:


#generating the independent and the dependent varaible 
x = data.drop(['Survived'], axis = 1 )
y = data['Survived']


# # scalling the data 

# In[10]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)


# In[16]:


from sklearn.model_selection import train_test_split 
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 96, stratify=y)


# # implementing KNN

# In[17]:


# importing KNN classifier and metric F1score 

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score 


# In[21]:


# creating instance of KNN
clf = KNN(n_neighbors=3)

# fiting the model 
clf.fit(train_x,train_y)

# predicting over the train set and calculating the F1 score 
train_predict = clf.predict(train_x)
k = f1_score(train_predict, train_y)
print('Training F1 score   ', k )

# predicting over the train set and calculating the F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 score ', k )


# # checking the traing f1 curve and test f1 curve 

# In[39]:


def F1score(k):
    '''
    takes an input K consisting of a range of K value for KNN
    input:
    k = list 
    Returns: lists containg f1 corresponding to every value f k 
    train_f1 = ist of train f1 score corresponding k 
    test_f1 = list of test f1 score corresponding to k 
    '''
    
    # initializing the empty list 
    train_f1 = []
    test_f1 = []
    
    
    # training model for every value of k 
    for i in k :
        # instance om KNN 
        clf = KNN(n_neighbors = i)
        clf.fit(train_x,train_y)
        # Appending F1 score to empty list calculated using the prediction
        tmp = clf.predict(train_x)
        tmp = f1_score(tmp, train_y)
        train_f1.append(tmp)
        
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp,test_y)
        test_f1.append(tmp)
    return train_f1, test_f1


# In[56]:


# Defining the range of K 
k = range(1,150)


# In[57]:


# calculating above defind function 
train_f1,test_f1 = F1score(k)


# In[58]:


score = pd.DataFrame({'train_score': train_f1, 'test_score':test_f1}, index = k)
score


# # Visualizing 

# In[59]:


plt.plot(k, test_f1, color= 'red',label = 'test')
plt.plot(k, train_f1, color = 'green', label = 'train')
plt.xlabel('K Neighbors')
plt.ylabel('F1 score')
plt.title('F1 Curve')
plt.ylim(0.4,1)
plt.legend()

