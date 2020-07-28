#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd


# In[153]:


df=pd.read_csv("train_bm.csv") 


# In[154]:


df.head()


# In[155]:


features = df.drop(['Item_Identifier','Item_Outlet_Sales'], axis = 1 )
target = df['Item_Outlet_Sales']


# In[156]:


from sklearn.neighbors import KNeighborsRegressor


# In[157]:


knn = KNeighborsRegressor()


# In[158]:


knn.fit(features, target) 


# In[159]:


from sklearn .metrics import accuracy_score


# In[160]:


df.isnull().sum()


# In[161]:


df['Item_Weight'].head(10)


# In[162]:


mean_val = df['Item_Weight'].mean()
mean_val


# In[163]:


df['Item_Weight'] = df['Item_Weight'].fillna(value = mean_val)


# In[164]:


df.isnull().sum()


# In[165]:


df['Outlet_Size'].head(10)


# In[166]:


mode_val = df['Outlet_Size'].mode()
mode_val


# In[167]:


df['Outlet_Size'] = df['Outlet_Size'].fillna(value=mode_val)


# In[168]:


df['Outlet_Size'].head()


# # categorical variables

# In[169]:


df.dtypes


# In[170]:


df['Outlet_Type'].value_counts()


# In[171]:


pd.get_dummies(df['Outlet_Type']).head()


# In[172]:


data = pd.get_dummies(df.drop(['Item_Identifier'], axis = 1 ))
data.head()


# # KNN classification 

# In[173]:


# importing the libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings 
warnings.filterwarnings('ignore')


# # load the data 

# In[174]:


data = pd.read_csv('data_cleaned.csv')
data.shape


# In[175]:


data.head()


# In[176]:


data.isnull().sum()


# In[177]:


data.dtypes


# In[178]:


x = data.drop(['Survived'], axis = 1)
y = data['Survived']
x.shape, y.shape 


# # Scaling the data (using MinMax scaler)

# In[179]:


# importiong the minmax scaler 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[180]:


x = pd.DataFrame(x_scaled, columns = x.columns)


# In[181]:


x.head()


# In[182]:


# importing the train test split function 
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56, stratify = y )


# # Implementing the KNN Classifier 

# In[183]:


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score


# In[184]:


# creating instance of KNN
clf = KNN(n_neighbors=5)

# fitting the model 
clf.fit(train_x,train_y)


# predicting over the main set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 Score',k )


# # Elbow for Classifier 

# In[185]:


def Elbow(k):
    #initiating empty list 
    test_error = []
    
    #training model for every value of k 
    for i in k:
        #instance on KNN
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        # appending the f1 score to the empty list calculated using the prediction 
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp, test_y)
        error = 1-tmp
        test_error.append(error)
    return test_error


# In[186]:


# Defining K range 
k = range(6, 20, 2)


# In[187]:


# calling the above defined function 
test = Elbow(k)


# In[188]:


# plotting the curves 
plt.plot(k, test)
plt.xlable('K Neighbors')
plt.ylable(' Test error')
plt.title('Elbow curve error')


# In[189]:


# creating instance of KNN
clf = KNN(n_neighbors=12)

# fitting the model 
clf.fit(train_x,train_y)


# predicting over the main set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 Score',k )


# # KNN Regression 

# In[190]:


data= pd.read_csv('train_cleaned.csv')
data.shape


# In[191]:


data.head()


# # segregating variable: Independent and Dependent Variable 

# In[192]:


x = data.drop(['Item_Outlet_Sales'], axis = 1)
y = data['Item_Outlet_Sales']
x.shape, y.shape 


# # Scaling all the value using the MinMax Scaler 

# In[193]:


# importing MinMax Scaler 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[194]:


x = pd.DataFrame(x_scaled)


# In[195]:


# importing train test split 
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# # importing the KNN regressor 

# In[196]:


# iporting the knn regresor and the metric use 

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import mean_squared_error as mse 


# In[ ]:


# creating instance of KNN 
reg = KNN(n_neighbors = 5)
 
# fitting the model 
reg.fit(train_x,train_y)

#predicting over the train set and calculating MSE
test_predict = reg.predict(test_x)
k = mse(test_predict, test_y)
print('Test MSE  ',k)


# In[197]:


df.dtypes


# In[198]:


def Elbow(k):
    #initiating empty list 
    test_mse = []
    
    #training model for every value of k 
    for i in k:
        #instance on KNN
        reg = KNN(n_neighbors = i)
        reg.fit(train_x, train_y)
        # appending the f1 score to the empty list calculated using the prediction 
        tmp = reg.predict(test_x)
        tmp = mse(tmp, test_y)
        test_mse.append(error)
    return test_mse


# In[199]:


#defining K range 
k = range(1,40)


# In[ ]:


test = Elbow(k)


# In[ ]:


# plotting the curve 
plt.plot(k, test)
plt.xlable('K Neighbors')
plt.ylable('Test Mean Squared error)
plt.title('Elbow Curve for test')


# In[ ]:


# creating instance of KNN 
reg = KNN(n_neighbors = 9)
 
# fitting the model 
reg.fit(train_x,train_y)

#predicting over the train set and calculating MSE
test_predict = reg.predict(test_x)
k = mse(test_predict, test_y)
print('Test MSE  ',k)

