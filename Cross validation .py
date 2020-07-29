#!/usr/bin/env python
# coding: utf-8

# # cross validation 

# In[10]:


# importing libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # importing the data 

# In[11]:


data = pd.read_csv('data_cleaned.csv')


# In[12]:


data.head()


# In[13]:


data.isnull().sum()


# # Segregating variables - Dependent & independent 

# In[14]:


# seprating independent and dependent variable
x = data.drop(['Survived'], axis = 1)
y = data['Survived']


# # scaling the data 

# In[21]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)


# In[22]:


from sklearn.model_selection import train_test_split
train_x, test_x,train_y,test_y = train_test_split(x,y, random_state = 96, stratify = y)


# # importing KNN

# In[23]:


# importing KNN classifier and metric F1score 
from sklearn.neighbors import KNeighborsClassifier as KNN


# # Checking Consistency, using Cross Validation 

# In[25]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(KNN(n_neighbors = 3), X = train_x, y = train_y, cv = 10 )
score 


# In[26]:


# consistency using mean and standard deviation in percentage 
score.mean()*100, score.std()*100


# # Automating the process of cross validation for diffrent K-Neighbors

# In[34]:


def Val_score(n_neighbors):
    """
    takes range of n_neighbors as input 
    returns Mean and standard Deviation for each value of n_neighbors 
    """
    
    avg = []
    std = []
    
    for i in n_neighbors:
        
        # 10 fold cross validation for every value of n_neighbor 
        score = cross_val_score(KNN(n_neighbors = i), X = train_x, y = train_y, cv = 10)
        
        # adding mean to average list 
        avg.append(score.mean())
        
        # adding standard deviation to std list 
        std.append(score.std())
        
    return avg, std 


# In[35]:


n_neighbors = range(1,50)
mean,std = Val_score(n_neighbors)


# # plotting mean validation score for each value of k 

# In[38]:


plt.plot(n_neighbors[10:20], mean[10:20] , color = 'green',label = 'mean')
plt.xlabel('n_neighbors')
plt.ylabel('Mean score')
plt.title('Mean Validation score')


# # plotting standard Deviation Validation Score for each K value 

# In[39]:


plt.plot(n_neighbors[10:20], std[10:20], color = 'red' , label = 'Standard deviation')
plt.xlabel('n_neighbors')
plt.ylabel('magnitute')
plt.title('Standard Deviation of Validation score ')


# # trying the optimal model over test_set 

# In[42]:


clf = KNN(n_neighbors = 14)
clf.fit(train_x, train_y)

score1 = clf.score(train_x, train_y)

score = clf.score(test_x, test_y)
score, score1


# In[ ]:




