#!/usr/bin/env python
# coding: utf-8

# # Hold _out_validation 

# In[1]:


# importing the libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # importing the data 

# In[4]:


data = pd.read_csv('data_cleaned.csv')
data.shape


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# # splitting 

# # Seprating dependent and independent variable 

# In[7]:


# for train_set
data_x = data.drop(['Survived'], axis = 1)
data_y = data['Survived']


# # creating Validation and test set 

# In[17]:


from sklearn.model_selection import train_test_split as tts 
train1_x,test_x , train1_y, test_y = tts(data_x, data_y , test_size = 0.2 , random_state = 50, stratify = data_y)


# In[18]:


train_x, val_x, train_y, val_y = tts(train1_x, train1_y, test_size = 0.2 , random_state = 51, stratify = train1_y)

print('training data   ',train_x.shape,train_y.shape)
print('validation data ',val_x.shape, val_y.shape)
print('test data       ',test_x.shape , test_y.shape)


# # checking Distribution of target class in train, test and validation set 

# In[19]:


train_y.value_counts()/len(train_y)


# In[20]:


val_y.value_counts()/len(val_y)


# In[21]:


test_y.value_counts()/len(test_y)


# In[ ]:




