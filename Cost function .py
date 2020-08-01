#!/usr/bin/env python
# coding: utf-8

# # cost creation
# 

# In[22]:


# import the ibraries 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse


# # creating sample data

# In[3]:


# creating the sample data set
experience = [1.2,1.5,1.9,2.2,2.4,2.5,2.8,3.1,3.3,3.7,4.2,4.4]
salary     = [1.7,2.4,2.3,3.1,3.7,4.2,4.4,6.1,5.4,5.7,6.4,6.2]

data = pd.DataFrame({
    "salary" : salary , 
    "Experience"  : experience
})
data.head()


# # plotting the grap 
# 

# In[5]:


#plotting the data 
plt.scatter(data.Experience , data.salary , color = 'red', label = 'data points')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()


# # Starting the line using small values of parameter
# 

# In[51]:


# making line for diffrent value of beta 0.1,0.8,1.5
beta = 0.10
# keeping intercept constant 
b = 1.1

#to store predictied points 
line1 = []

# generating prediction for every data points 
for i in range(len(data)):
    line1.append(data.Experience[i]*beta + b)
    

    
# plotting the line 
plt.scatter(data.Experience, data.salary , color = 'red')
plt.plot(data.Experience, line1 , color = 'black',label = 'line')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
MSE = mse(data.Experience,line1)
plt.title("Beta value"+str(beta)+" with MSE " + str(MSE))
MSE = mse(data.Experience, line1)


# # computing Cost over a range of value of Beta 
# 

# In[20]:


# function to calculating error 
def Error(Beta, data):
    #b is  constant 
    b = 1.1 
    
    salary = []
    Experience = data.Experience
    
    # loop to calculate predict salary variable 
    for i in range(len(data.Experience)):
        tmp = data.Experience[i] * Beta + b 
        salary.append(tmp)
    MSE = mse(Experience, salary)
    return MSE


# In[32]:


# range of slopes from 0 to 1.5 with increment of 0.01
slope = [i/100 for i in range(0, 150)]
Cost = []
for i in slop:
    cost = Error( Beta = i, data = data)
    Cost.append(cost)
    


# In[33]:


# Arranging in data frame
Cost_table = pd.DataFrame({
    'Beta' : slope,
    'Cost' : Cost
})
Cost_table.head()


# # Visiualising cost with respect to beta 

# In[38]:


# plotting the cost value corresponding to every value of Beta
plt.plot(Cost_table.Beta, Cost_table.Cost, color = 'blue', label = 'Cost function Curve')
plt.xlabel('Value of Beta')
plt.ylabel('Cost')
plt.legend()

