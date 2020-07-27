#!/usr/bin/env python
# coding: utf-8

# # building first predictive model 

# # Importing the libraries 

# In[33]:


import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


# # Importing data sheet 

# In[34]:


data = pd.read_csv('train_bm.csv')


# In[35]:


data.shape


# In[36]:


data.head()


# In[37]:


data.isnull().sum()


# # shuffling and creating train and test set 

# In[38]:


from sklearn.utils import shuffle 

#Shuffling the data set 
data = shuffle(data,random_state = 42 )

# creating 4 division 
div = int(data.shape[0]/4)

# 3 part to train set and one part to test set 
train = data.loc[:3*div+1,:]
test = data.loc[3*div+1:]


# In[39]:


train.head()


# In[40]:


test.head()


# # simple mean model(mean of item_outlet_sales )

# In[41]:


# storing simple mean in the new column in the test set as "simple_mean"
test['simple_mean'] = train['Item_Outlet_Sales'].mean()


# In[42]:


# calculating mean absolute error 
from sklearn.metrics import mean_absolute_error as MAE

simple_mean_error = MAE(test['Item_Outlet_Sales'], test['simple_mean'])
simple_mean_error


# # Mean item Outlet Sales with respect to Outlet_types 

# In[43]:


out_type = pd.pivot_table(train, values = 'Item_Outlet_Sales', index = ['Outlet_Type'], aggfunc=np.mean)
out_type


# In[44]:


# initializing new column to zero 
test['Out_type_mean'] = 0 

#for every unique entry in outlet identifier 
for i in train['Outlet_Type'].unique():
    # Assign the mean value corresponding to unique vaue 
    test['Out_type_mean'][test['Outlet_Type'] == str(i)] = train ['Item_Outlet_Sales'][train['Outlet_Type'] == str(i)].mean()


# In[45]:


out_type_error = MAE(test['Item_Outlet_Sales'], test['Out_type_mean'])
out_type_error


# In[46]:


out_year = pd.pivot_table(train, values='Item_Outlet_Sales', index=['Outlet_Establishment_Year'], aggfunc=np.mean)
out_year


# # Mean_item Outlet sales with respect to outlet_Establishment_year

# In[47]:


out_year = pd.pivot_table(train, values ='Item_Outlet_Sales', index = ['Outlet_Establishment_Year'], aggfunc=np.mean)
out_year


# In[48]:


# initializing new column to zero
test['Out_year_mean'] = 0 

#for every unique entry in output identifiers 
for i in train['Outlet_Establishment_Year'].unique():
    #  assign the main value corrosponding to new entry 
    test['Out_year_mean'][test['Outlet_Establishment_Year'] == str(i)] = train['Item_Outlet_Sales'][train['Outlet_Establishment_Year']== str(i)].mean()


# In[49]:


#calculating mean absolute error
out_year_error = MAE(test['Item_Outlet_Sales'], test['Out_year_mean'])
out_year_error


# # mean item Outlet sales with respect to outlet_location_Type

# In[50]:


out_loc = pd.pivot_table(train, values ='Item_Outlet_Sales', index = ['Outlet_Location_Type'], aggfunc=np.mean)
out_loc


# In[51]:


#initializing new column to zero
test['Out_loc_mean'] = 0 

#for every unique entry in output identifiers 
for i in train['Outlet_Location_Type'].unique():
    #  assign the main value corrosponding to new entry 
    test['Out_loc_mean'][test['Outlet_Location_Type'] == str(i)] = train['Item_Outlet_Sales'][train['Outlet_Location_Type']== str(i)].mean()


# In[52]:


#calculating mean absolutr error 
out_loc_error = MAE(test['Item_Outlet_Sales'], test['Out_loc_mean'])
out_loc_error


# # Mean Item_Outlet_Sales with respect to both Outlet_Location_Type and Outlet_Establishment_year

# In[53]:


combo = pd.pivot_table(train, values ='Item_Outlet_Sales', index = ['Outlet_Location_Type','Outlet_Establishment_Year'], aggfunc=np.mean)
combo


# In[55]:


#initializing new column to zero
test['Super_mean'] = 0 

# assuming variable to string (to shorten code lenght)
s2 = 'Outlet_Location_Type'
s1 = 'Outlet_Establishment_Year'
#for every unique value in s1
for i in test[s1].unique():
    # for every uniuque value in s2
    for j in test[s2].unique():  
        test['Super_mean'][(test[s1] == i)& (test[s2]==str(j))] = train['Item_Outlet_Sales'][(train[s1]== i) & (train[s2]== str(j))].mean()


# In[56]:


out_loc_error = MAE(test['Item_Outlet_Sales'], test['Super_mean'])
out_loc_error


# In[ ]:




