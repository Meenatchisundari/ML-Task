#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


dataset=pd.read_csv("C://Users//ASUS//Downloads//Person will buy a car or not.csv")
print("Dataset imported successfully")


# In[4]:


dataset


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


dataset.shape


# In[8]:


dataset.info()


# In[9]:


dataset.isnull().sum()


# In[10]:


dataset.describe()


# In[11]:


dataset['target'].value_counts()


# In[12]:


X=dataset.drop(columns='target',axis=1)
Y=dataset['target']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[16]:


print(X.shape,X_train.shape,X_test.shape)


# In[17]:


model=LogisticRegression()


# In[18]:


model.fit(X_train,Y_train)


# In[19]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[20]:


print('Accuracy on Training data:',training_data_accuracy)


# In[21]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[22]:


print('Accuracy on Test data:',test_data_accuracy)


# In[24]:


input_data=(48,0,2,130,275,0,1,139,0,0.2,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person will not buy a car")
else:
    print("The person will buy a car")


# In[25]:


input_data=(59,1,0,170,326,0,0,140,1,3.4,0,0,3)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person will not buy a car")
else:
    print("The person will buy a car")


# In[ ]:




