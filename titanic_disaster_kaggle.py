#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd


# In[115]:


data=pd.read_csv(r"C:\\Users\\admin\Desktop\train.csv")


# In[116]:


data.head(n=5)


# In[80]:


data.info()


# In[160]:


columns_to_drop=["Name","Ticket","Cabin","Embarked"]
data_clean=data.drop(columns_to_drop,axis=1)
data_clean.head(n=5)


# In[161]:


print(type(data_clean))


# In[162]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[163]:


data_clean["Sex"]=le.fit_transform(data_clean["Sex"])


# In[164]:


data_clean.head(n=5)


# In[165]:


print(type(data_clean))


# In[166]:


data_clean.info()


# In[167]:


data_clean=data_clean.fillna(data_clean["Age"].mean())


# In[168]:


data_clean.info()


# In[169]:


print(type(data_clean))


# In[181]:


input_cols=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']
output_cols=['Survived']
             
X = data_clean[input_cols]
Y = data_clean[output_cols]             


# In[182]:


print(X)


# In[183]:


split=int(0.7*data_clean.shape[0])
train_data=data_clean[:split]
test_data=data_clean[split:]
test_data=test_data.reset_index(drop=True)


# In[184]:


from sklearn.tree import DecisionTreeClassifier


# In[185]:


sk_tree=DecisionTreeClassifier(criterion='entropy',max_depth=5)


# In[186]:


sk_tree.fit(train_data[input_cols],train_data[output_cols])


# In[187]:


print(train_data[output_cols])


# In[205]:


#y_pred=sk_tree.predict(test_data[input_cols])
#print(y_pred)
#print(y_pred.shape)


# In[206]:


sk_tree.score(test_data[input_cols],test_data[output_cols])


# In[207]:


test=pd.read_csv(r"C:\\Users\\admin\Desktop\test.csv")


# In[208]:


san=test[input_cols]
san


# In[209]:


san["Sex"]=le.fit_transform(san["Sex"])
san=san.fillna(san["Age"].mean())


# In[210]:


san.head()


# In[211]:


san.info()


# In[212]:


y_pred=sk_tree.predict(san)
print(y_pred)
print(y_pred.shape)


# In[214]:


submission=pd.DataFrame({
    "PassengerId":san["PassengerId"],"Survived":y_pred
})
submission.to_csv('submission.csv',index=False)


# In[215]:


submission.head()


# In[ ]:




