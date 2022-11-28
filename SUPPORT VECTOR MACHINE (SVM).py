#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[5]:


iris.keys()


# In[10]:


iris.data


# In[13]:


iris.target


# In[15]:


iris.feature_names


# In[16]:


print(iris.DESCR)


# In[27]:


iris = pd.DataFrame(np.c_[iris["data"],iris["target"]],columns=np.append(iris["feature_names"],["target"]))
iris


# In[28]:


x = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values


# In[29]:


iris.isna().sum()


# In[33]:


sns.countplot(iris["target"])


# In[40]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[41]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[42]:


x_train


# In[43]:


x_test


# In[47]:


from sklearn.svm import SVC


# In[50]:


classifier=SVC(kernel="rbf")
classifier.fit(x_train,y_train)


# In[51]:


y_pred=classifier.predict(x_test)


# In[61]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[62]:


sns.heatmap(cm,annot=True)


# In[65]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac


# In[ ]:




