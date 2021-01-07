#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ### Imports we will be using for this pipeline

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# ### Imputing in the Pipeline 
# #### Put together a loading function to access the data from a remote site instead of from classical libraries with the regular import. Will use this data as an imputer during the pipeline

# In[21]:


def load_heart_disease():
    '''
    Description: 
        Load and pre-process heart disease data
        If processed.hungarian.data file is not present it will be downloaded from:
        https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data
    
    Input: 
        none
    
    Returns: 
        data(DataFrame)
    
    '''
    
    import os
    import requests
    
    
    file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
    file_name = file_url.split('/')[-1]
    
    if not os.path.isfile(file_name):
        print('Downloading from {}'.format(file_url))
        r = requests.get(file_url)
        with open(file_name,'wb') as output_file:
            output_file.write(r.content)
        
    data = pd.read_csv(file_name, 
                   na_values='?', 
                   names=[ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                            'ca', 'thal', 'num'])
    
    
    return data


# ### Column Data Types
# #### Taking a look at our data to see which columns have the most entries, which are less reliable and the variable types that we’ll be working with. Glad to only have floats and integers. If string variable types existed then we would’ve make them categorical similar to how sex is in 0's and 1’s.

# In[22]:


data = load_heart_disease()


# In[23]:


#Initial glance at data
data.head()


# In[24]:


#Checking column data types
data.info()


# ### Data Conversion
# #### Converted everything to a float and then all viable columns back to Int’s to allow for future OneHotEncoder and scalar conversions

# In[25]:


#Converting all columns to float
data = data.astype('float64')

#Categorical back to int
#Using pandas Int64 to keep missing values
data = data.astype({col: 'Int64' for col in [ 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']})
data.info()


# ### Data Inspection cont.
# #### Inspect the composition of our columns to identify numeric and categorical features

# In[26]:


#Looking at the types of columns we have
{col: data[col].unique() for col in data.columns}


# ### Preprocessing and Pipeline generation
# #### Break apart our two types of features to apply their own preprocessing. Numeric features did not display exponential characteristics so a simple median scaling can be used to fill in missing data. Categorical features with missing data were filled in with their most common missing variable of that feature. One hot encoding was used so that values are either yes or no and it gives our model a stronger breakdown of our data features

# In[27]:


#Setting up the pipeline

#For numeric features we impute with median and apply standard scaling.
#For categorical features we imput with most frequent and apply one-hot encoding.

#numeric features
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
numeric_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])

#categorical features
categorical_features = [ 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
categorical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#numeric and categorical transformers applied
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#Pipeline set up with preprocessor and classifier
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000))])


# ### Data split
# #### Splitting the data and confirming the shapes match

# In[28]:


#split the data
#get all columns apart from income for the features
X = data.drop(columns='num')
y = data['num']
print(X.shape)
print(y.shape)

# split dataframe and income
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1, 
                                                    stratify=y,
                                                    random_state=31)


# ### Grid Search for Best Parameters
# #### We manually defined the parameter grid for the grid search to iterate through with some fairly normal inputs. If the C parameter ends on either of the outlying extremes then it would be wise to try a new set of parameters shifted towards that extreme.

# In[29]:


#grid search
param_grid = {'classifier__C': [0.1, 1.0, 10.0, 100.0],
             'classifier__fit_intercept': [True, False]}
grid = GridSearchCV(pipe, 
                    param_grid, 
                    cv=5, 
                    return_train_score=True)

grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation train score: {:.2f}".format(grid.cv_results_['mean_train_score'][grid.best_index_]))
print("Best cross-validation test score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))


# ### Model Visualisation
# #### Output some of the key characteristics of our processed data. Including the beginning features names and the pipeline they are associated with, their processed column with OneHotEncoder, and a visualization of their weights within the model.

# In[30]:


#Visualizing Coefficients
print(grid.best_estimator_['preprocessor'])

#output all processed columns
processed_feature_names = numeric_features +                        grid.best_estimator_['preprocessor'].transformers_[1][1]['onehot']                           .get_feature_names(categorical_features).tolist()
print(processed_feature_names)

#plot the magnitudes of the features for a better understanding on their influence on the model
plt.plot(grid.best_estimator_['classifier'].coef_.T, 'o')

plt.xticks(range(len(processed_feature_names)), processed_feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
# plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
# plt.legend()


# ### Revisiting with a second model
# #### Remade the same pipeline from above for easy reference

# In[31]:


#pipeline for best heart disease model

#encoded preprocessor using imputing
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
numeric_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])

categorical_features = [ 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
categorical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
                                    ('num', numeric_transformer, numeric_features),
                                    ('cat', categorical_transformer, categorical_features)])


# ### Imputer and Pipeline creation
# #### Created the imputer for prepoccessing

# In[32]:


#un-encoded preprocessing (using imputing only)
imputer = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)])

#Append classifier to preprocessing pipeline.
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000))])


# ### Defining Grid Search Parameters
# #### Set up the parameters to test for a Logistic Regression and Random Forest Classifier

# In[33]:


#setting up grid search parameter grid
param_grid = [
    {'classifier': [LogisticRegression(max_iter=5000)], 
         'preprocessor': [preprocessor, imputer],
         'classifier__C': [0.1, 1.0, 10.0, 100.0],
         'classifier__fit_intercept': [True, False]},
    {'classifier': [RandomForestClassifier(random_state=58)],
        'preprocessor': [preprocessor, imputer], 
        'classifier__n_estimators': [20, 50, 100, 200],
        'classifier__max_depth': [1, 2, 3, 5, 7],
        'classifier__max_features': ['auto', 'log2', None]}]

grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True)


# ### Data Split
# #### Splitting the data for training

# In[34]:


#split data
X = data.drop(columns='num')
y = data['num']
print(X.shape)
print(y.shape)

# split dataframe and income
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.1, stratify=y,random_state=31)


# ### GridSearch
# #### Conducting the gridsearch and outputting the best mode alongside its best parameters

# In[35]:


grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation train score: {:.2f}".format(grid.cv_results_['mean_train_score'][grid.best_index_]))
print("Best cross-validation test score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))


# ### Dumping Model
# #### Saving the model to joblib for potential future use

# In[36]:


#save the model for future use
from joblib import dump, load
dump(grid.best_estimator_, 'heart-disease-classifier.joblib')


# ### Checking Model Dump
# #### Test loading of the model

# In[37]:


#confirm that model can be loaded
clf = load('heart-disease-classifier.joblib') 
print("Test-set score: {:.2f}".format(clf.score(X_test, y_test)))
clf


# In[ ]:





# In[ ]:




