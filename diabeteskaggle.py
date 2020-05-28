#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[130]:


x1=pd.read_csv("C:\\Users\\admin\\Desktop\\Datasets\\diabetes knn\\Diabetes_XTrain.csv")
y1=pd.read_csv("C:\\Users\\admin\\Desktop\\Datasets\\diabetes knn\\Diabetes_YTrain.csv")


# In[131]:


print(y1.shape)
print(x1.shape)


# In[132]:


x1.head()


# In[133]:


"""to check if the dataset has any missing value"""
x1.isnull().values.any()


# In[134]:


"""to check which all columns in our training dataset have zero values,we will have to replace them"""
print("no of zeros in Pregnancies:{0}".format(len(x1.loc[x1['Pregnancies']==0])))
print("no of zeros in Glucose:{0}".format(len(x1.loc[x1['Glucose']==0])))
print("no of zeros in BloodPressure:{0}".format(len(x1.loc[x1['BloodPressure']==0])))
print("no of zeros in SkinThickness:{0}".format(len(x1.loc[x1['SkinThickness']==0])))
print("no of zeros in Insulin :{0}".format(len(x1.loc[x1['Insulin']==0])))
print("no of zeros in BMI:{0}".format(len(x1.loc[x1['BMI']==0])))
print("no of zeros in DiabetesPedigreeFunction:{0}".format(len(x1.loc[x1['DiabetesPedigreeFunction']==0])))
print("no of zeros in Age  :{0}".format(len(x1.loc[x1['Age']==0])))


# In[135]:


"""removing the zeroes"""


# In[136]:


x=x1.values
y=y1.values


# In[161]:


"""PLOTTING A BAR GRAPH"""
import seaborn as sns
sns.countplot(y1['Outcome'],label="count")


# In[137]:


from sklearn.model_selection import train_test_split


# In[138]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[139]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[140]:


#this is the method to remove the zeroes with the mean values


# In[141]:


from sklearn.impute import SimpleImputer
fill_values=SimpleImputer(missing_values=0,strategy="mean")
x_train=fill_values.fit_transform(x_train)
x_test=fill_values.fit_transform(x_test)


# In[142]:


"""Method 2 to remove the zeros(but you will have to use it before you define train_test_split)"""
#column_not_accepted=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']
#for column in column_not_accepted:
    #x1[column]=x1[column].replace(0,np.NaN)
    #mean=int(x1[column].mean(skipna=True))
    #x1[column]=x1[column].replace(np.NaN,mean)


# In[143]:


"""we will have to scale our dataor the features(this helps us to increase the accuracy)"""


# In[144]:


from sklearn.preprocessing import StandardScaler


# In[145]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)


# In[146]:


x_train


# In[147]:


"""Now we will have to use knn for predictions"""


# In[148]:


#to assume the value of k or n_neighbors use
import math
math.sqrt(len(y_test))


# In[149]:


classifier=KNeighborsClassifier(n_neighbors=9,p=2,metric='euclidean')
classifier.fit(x_train,y_train.ravel())


# In[150]:


print(x_test.shape)


# In[151]:


#now we predict
y_pred=classifier.predict(x_test)
y_pred


# In[152]:


print(y_test.shape)
print(y_pred.shape)
print(x_test.shape)


# In[153]:


y_pred=y_pred.reshape(y_test.shape[0])


# In[154]:


"""ACCURACY"""
print(accuracy_score(y_test,y_pred))


# In[162]:


submission=pd.DataFrame({
    "Outcome":y_pred
})
submission.to_csv('submission_diabetes.csv',index=False)


# In[ ]:




