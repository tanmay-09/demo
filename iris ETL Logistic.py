#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A = pd.read_csv("iris-with-answers.csv")


# In[2]:


A


# In[3]:


copy_data=A.copy()


# In[4]:


A['species']=A['species'].map({'setosa':0,'versicolor':1,'virginica':2})


# In[5]:


A


# In[6]:


X = A[["sepal_length","sepal_length","petal_length","petal_width"]]
Y = A[["species"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=35)


# repeat this stament if your model is having sampling bias

from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
model = lm.fit(xtrain,ytrain)
b0 = model.intercept_
b1 = model.coef_
pred = model.predict(xtest)

ytest['predict']=pred
print(ytest)
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
print(mean_absolute_error(ytest.species,pred))
print(mean_squared_error(ytest.species,pred))
print(explained_variance_score(ytest.species,pred))


# In[11]:



import pickle

pickle.dump(lm,open("iris.pkl","wb"))
model=pickle.load(open('iris.pkl','rb'))


# In[ ]:





# In[ ]:




