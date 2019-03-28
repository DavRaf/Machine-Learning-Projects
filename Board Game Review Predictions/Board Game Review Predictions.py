
# coding: utf-8

# In[2]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


# Load the data
games = pd.read_csv("games.csv")


# In[5]:


# Print the names of the columns in games
print(games.columns)
print(games.shape)
games.describe()


# In[6]:


# Make a histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])
plt.show()


# In[7]:


# Print the first row of all the games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

# Print the first row of games with scores grater than 0
print(games[games["average_rating"] > 0].iloc[0])


# In[10]:


# Remove any rows without user reviews
games = games[games['users_rated'] > 0]

# Remove any rows with missing values
games = games.dropna(axis=0)

# Show the shape of the data
print(games.shape)

# Make a histogram of all the avrage ratings
plt.hist(games["average_rating"])
plt.show()


# In[12]:


# Correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax= .8, square = True)
plt.show()


# In[13]:


# Get all the columns from the dataframe
columns = games.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the varibale we'll be predicting on
target = "average_rating"


# In[14]:


# Generate training and test datasets
from sklearn.model_selection import train_test_split

# Generate the training set
train = games.sample(frac=0.8, random_state = 1)

# Select anything not in the training set and put it in test
test = games.loc[~games.index.isin(train.index)]

# Print shapes
print(train.shape)
print(test.shape)


# In[22]:


# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
linear_regression_model = LinearRegression()

# Fit the model the training data
linear_regression_model.fit(train[columns], train[target])


# In[23]:


# Generate predictions for the test set
predictions = linear_regression_model.predict(test[columns])

# Compute error between our test predictions and actual values
mean_squared_error(predictions, test[target])


# In[24]:


# Import the random forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
random_forest_regression_model = RandomForestRegressor(n_estimators = 100, min_samples_leaf=10, random_state=1)

# Fit the data
random_forest_regression_model.fit(train[columns], train[target])


# In[25]:


# Generate predictions for the test set
predictions = random_forest_regression_model.predict(test[columns])

# Compute error between our test predictions and actual values
mean_squared_error(predictions, test[target])


# In[28]:


test[columns].iloc[5]


# In[31]:


# Make predictions with both models
rating_LR = linear_regression_model.predict(test[columns].iloc[50].values.reshape(1, -1))
rating_RFR = random_forest_regression_model.predict(test[columns].iloc[50].values.reshape(1, -1))

# Print out the predictions
print(rating_LR)
print(rating_RFR)


# In[32]:


test[target].iloc[50]

