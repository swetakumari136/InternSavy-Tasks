#!/usr/bin/env python
# coding: utf-8

# In[76]:


# Import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import timeit
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# In[77]:


# Import dataset.
df = pd.read_csv("Admission_Predict.csv",sep = ",")
df.head()


# In[78]:


# Check for null values.
df.isnull().sum()

# Drop 'Serial No.'
df = df.drop('Serial No.',axis=1)

# Renaming columns.
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ','_')


# In[79]:


df.columns


# In[80]:


# Check if the data types are correct as per the meaning of each column.
df.info()


# In[81]:


# Descriptive statistics.
df.describe()


# In[82]:


cols = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Chance_of_Admit']

plt.figure(figsize=(6,40))

for i in range(len(cols)):
    plt.subplot(7,1,i+1)
    plt.hist(df[cols[i]],color='pink',alpha=0.75)
    plt.title("Distribution of " + cols[i])

plt.show()


# In[83]:


cols = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA']

plt.figure(figsize=(6,40))

for i in range(len(cols)):
    plt.subplot(6,1,i+1)
    plt.scatter(df['Chance_of_Admit'],df[cols[i]],color='brown')
    plt.title("Correlation b/w 'Chance_of_Admit' and '{}'".format(cols[i]))

plt.show()


# In[84]:


# Visualize correlation between independant variables and the target variable. Here, the target variable is 'Chance_of_Admit'
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[85]:



df=df.rename(columns = {'Chance_of_Admit':'Chance_of_Admit'})
X=df.drop('Chance_of_Admit',axis=1)
y=df['Chance_of_Admit']


# In[86]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[87]:


#If Chance of Admit greater than 80% we classify it as 1
y_train_c = [1 if each > 0.8 else 0 for each in y_train]
y_test_c  = [1 if each > 0.8 else 0 for each in y_test]


# In[88]:


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,mean_squared_error


# In[89]:


classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gausian Naive Bayes :',GaussianNB()]]


# In[90]:


cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train_c)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test_c,predictions))
    print(name,accuracy_score(y_test_c,predictions))


# In[91]:


y_ax=['Logistic Regression' ,
      'Decision Tree Classifier',
      'Random Forest Classifier',
      'Gradient Boosting Classifier',
      'Ada Boosting Classifier',
      'Extra Tree Classifier' ,
      'K-Neighbors Classifier',
      'Support Vector Classifier',
       'Gaussian Naive Bayes']
x_ax=cla_pred
sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.8")
plt.xlabel('Accuracy')


# In[ ]:




