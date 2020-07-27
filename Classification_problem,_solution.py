#!/usr/bin/env python
# coding: utf-8

#  # classification Benchmark
#  

# In[2]:


# improrting the libraries 
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score


# # Importing Dataset 

# In[3]:


data = pd.read_csv('train1.csv')
data.shape


# In[26]:


data.head()


# In[27]:


data.isnull().sum()


# # shuffling and creating train and Test set 

# In[9]:


from sklearn.utils import shuffle 

# shuffling the dataset 
data = shuffle(data, random_state = 42)

#creating 4 division 
div = int(data.shape[0]/4)

# 3 part to train set and one part to test set 
train = data.loc[:3*div+1,:]
test = data.loc[3*div+1:]

train.shape,test.shape


# In[10]:


train.head()


# In[11]:


test.head()


# # simple mode 

# In[12]:


test['simple_mode'] = train['Survived'].mode()[0]
test['simple_mode'].head()


# In[14]:


simple_mode_accuracy = accuracy_score(test['Survived'], test['simple_mode'])
simple_mode_accuracy


# # mode based on the gender 

# In[17]:


gender_mode = pd.crosstab(train['Survived'],train['Sex'])
gender_mode


# In[20]:


test['gender_mode'] = test['Survived']

#for every unique value in column 
for i in test['Sex'].unique():
    # calculate and assign mode to new column corresponding to the unique value "Sex"
    test['gender_mode'][test['Sex']== str(i)] = train['Survived'][train['Sex']== str(i)].mode()[0]


# In[24]:


gender_accuracy = accuracy_score(test['Survived'], test['gender_mode'])
gender_accuracy*100


# In[31]:


class_mode = pd.crosstab(train['Survived'],train['Pclass'])
class_mode

