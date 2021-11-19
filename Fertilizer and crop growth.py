#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset=pd.read_excel("C://Users//ASUS//Downloads//Fertilizer vs crop growth.xlsx")
print("Dataset imported successfully")


# In[3]:


dataset


# In[4]:


dataset.shape


# In[5]:


dataset.head()


# In[6]:


dataset.describe()


# In[7]:


dataset.plot(x='Fertilizer in ppm',y='Crop growth in percentage',style='o')
plt.title("Fertilizer vs Crop growth")
plt.xlabel("Amount of fertilizer in ppm")
plt.ylabel("Crop growth in percentage")
plt.show()


# In[8]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("Training complete.")


# In[11]:


print(regressor.intercept_)
print(regressor.coef_)


# In[12]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[13]:


print(x_test)
y_pred=regressor.predict(x_test)


# In[14]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[15]:


from sklearn import metrics
print("Mean Absolute Error=",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error=",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error=",metrics.mean_squared_error(y_test,y_pred))


# In[16]:


Fertilizer =9.25
test=np.array([Fertilizer])
test=test.reshape(-1,1)
own_pred=regressor.predict(test)
print("Amount of fertilizer in ppm={}".format(Fertilizer))
print("Predicted Crop growth in percentage={}".format(own_pred[0]))

