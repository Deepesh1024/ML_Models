#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt 


# In[2]:


housing = pd.read_csv("/Users/deepeshjha/Desktop/DSnML/housingdata.csv",encoding='windows-1254')
housing.head()


# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


# In[6]:


housing.hist(bins=50,figsize=(20,15))


# # Train Test splitting

# In[7]:


#for learning process
# def split_train_test(data,test_ratio):
#     shuffled = np.random.permutation(len(data))
#     np.random.seed(42)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


#train_set, test_set = split_train_test(housing,0.2)


# In[9]:


#print(f"Rows in train test : {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[10]:


housing['CHAS'].value_counts()


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size= 0.2,random_state=42)


# In[12]:


print(f"Rows in train test : {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split =  StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


housing = strat_train_set.copy()


# ## Looking for CoRelations 

# In[16]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[17]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[18]:


housing.plot(kind = "scatter", x = "RM", y = "MEDV", alpha = 0.8)


# ## Trying out Attribute combinations

# In[19]:


housing["TAXRM"] = housing['TAX']/housing['RM']
housing["TAXRM"]


# In[20]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[21]:


housing.plot(kind="scatter", x = "TAXRM", y = "MEDV", alpha = 0.8)


# In[22]:


housing = strat_train_set
housing.drop(columns="MEDV")
housing_labels = strat_train_set['MEDV'].copy()


# ## Missing Attributes

# In[23]:


# To take care of the missing attributes, you have three options:
# 1. Get rid of the missing data points
# 2. Get rid of whole attribute
# 3 . Set the value to same(0,mean or median)


# In[24]:


# a = housing.dropna(subset=["RM"]) #Option 1
# a.shape


# In[25]:


# #To delete a column #Option 2
# housing.drop("RM", axis = 1)
# #Now the RM coplumn will be removed and also note that original housing data frame will remain unchanged


# In[26]:


# #Option 3
# median = housing["RM"].median()
# median
# housing["RM"].fillna(median)
# Original dataFrame is remained unchanged


# In[27]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[28]:


imputer.statistics_


# In[29]:


x = imputer.transform(housing)


# In[30]:


housing_tr = pd.DataFrame(x , columns=housing.columns)


# In[31]:


housing_tr.describe()


# In[32]:


housing.shape


# ## Scikit-Learn Design 

# Primarly , three types of objects
# 
# 1.Estimators - It estimates some parametersnbased on a dataset. Eg.Imputer
# It has a fit method and transform method
# Fit menthod - Fits the dataset and calculates internal parameters
# 
# 2.Transformers - Transform method takes input and returns output based on the learnings from fit().
# It also has a convienience fucntion called fit_transform() which fits and then transforms.
# 
# 3.Predictors - LogisticRegression model is an example of predictor . fit() and predict() are two common fucntions.
# It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily, two types of feature scaling methods:
# 
# 1.Min-Max Scaling (normalization)
# (value-min)/(max-min)
# SkLearn provides a class called MinMaxScaler for this
# 
# 2.Standardization
# (value-mean)/std 
# Sklearn provides a class called StandardScaler for thus

# ## Creating  a pipeline 

# In[33]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[34]:


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[35]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[36]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = DecisionTreeRegressor()
# model = LinearRegression()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[38]:


#model.predict(strat_test_set)


# In[39]:


some_data = housing.iloc[:5]


# In[40]:


some_labels = housing_labels.iloc[:5]


# In[41]:


prepared_data = my_pipeline.transform(some_data)


# In[42]:


model.predict(prepared_data)


# In[43]:


list(some_labels)


# ## Evaluating the model 

# In[44]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions )
np.sqrt(mse)


# ## Using better evaluation technique - Cross Validation

# In[45]:


from sklearn.model_selection import cross_val_score
scores  = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[46]:


rmse_scores


# In[47]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


# In[48]:


print_scores(rmse_scores)

