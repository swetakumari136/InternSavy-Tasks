#!/usr/bin/env python
# coding: utf-8

# In[65]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[39]:


#Import Dataset
customer=pd.read_csv("Mall_Customers.csv")


# In[40]:


customer.head()


# In[56]:


customer.shape


# In[57]:


# Check null values
customer.isnull().sum()


# In[58]:


# Rename Columns 
customer.rename(columns = {'Annual Income (k$)':'Annual_Income_(k$)' , 'Spending Score (1-100)': 'Spending_Score'}, inplace = True)
customer.head()


# In[60]:


# Spending_Score mean with respect to Genre
customer[['Spending_Score',  'Genre']].groupby(['Genre']).mean()


# In[41]:


customer.columns


# In[42]:


customer.describe


# In[43]:



# See the distribution of gender to recognize different distributions
sns.countplot(x='Genre', data=customer);
plt.title('Distribution of Genre');


# In[44]:


# Create a histogram of ages
customer.hist('Age', bins=35);
plt.title('Distribution of Age');
plt.xlabel('Age');


# In[45]:


plt.hist('Age', data=customer[customer['Genre'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Age', data=customer[customer['Genre'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Age by Genre');
plt.xlabel('Age');
plt.legend();


# In[46]:


customer.hist('Annual Income (k$)');
plt.title('Annual Income Distribution in Thousands of Dollars');
plt.xlabel('Thousands of Dollars');


# In[47]:


# Histogram of income by gender
plt.hist('Annual Income (k$)', data=customer[customer['Genre'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Annual Income (k$)', data=customer[customer['Genre'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Income by Genre');
plt.xlabel('Income (Thousands of Dollars)');
plt.legend();


# In[48]:


# Create data sets by gender to save time in the future since gender seems to significantly impact other variables
male_customers = customer[customer['Genre'] == 'Male']
female_customers = customer[customer['Genre'] == 'Female']

# Print the average spending score for men and women
print(male_customers['Spending Score (1-100)'].mean())
print(female_customers['Spending Score (1-100)'].mean())


# In[49]:


sns.scatterplot('Age', 'Annual Income (k$)', hue='Genre', data=customer);
plt.title('Age to Income, Colored by Gender')


# In[50]:


sns.heatmap(female_customers.corr(), annot=True);
plt.title('Correlation Heatmap - Female');


# In[51]:


sns.lmplot('Age', 'Spending Score (1-100)', data=female_customers);
plt.title('Age to Spending Score, Female Only');


# In[52]:


sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', hue='Genre', data=customer);
plt.title('Annual Income to Spending Score, Colored by Genre');


# In[69]:


X = customer[['Annual Income(k$)', 'Spending Score(1-100)']]


# In[ ]:




