#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


news=pd.read_csv("C://Users//ASUS//OneDrive//Documents//news.csv")


# In[3]:


print("Dataset imported successfully")


# In[4]:


news


# In[5]:


labels=news.label
labels


# In[6]:


news['text']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split( news['text'],labels, test_size=0.20, random_state=0)


# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(stop_words='english',max_df=0.25)
tfidf_train=tfidf_vect.fit_transform(X_train)
tfidf_test=tfidf_vect.transform(X_test)


# In[13]:


from sklearn.linear_model import SGDClassifier
fake_detector_svc = SGDClassifier().fit(tfidf_train, Y_train)


# In[14]:


predictions =fake_detector_svc.predict(tfidf_test)
predictions


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))


# In[ ]:




