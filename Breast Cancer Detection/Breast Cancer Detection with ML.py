
# coding: utf-8


# In[3]:


import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix 


# In[10]:


# Loading the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(filepath_or_buffer=url, names=names) # creates a dataframe having the specified data (indicated in the url) and headers (indicated in the names)


# In[11]:


# Preprocess the data
df.replace(to_replace='?', value=-99999, inplace = True) # replaces all ? with -99999
print(df.axes) # gives information about the dataset (number of the observations and the variables)

df.drop(labels=['id'], axis=1, inplace=True) # drops a variable

# Print the shape of the dataset
print(df.shape) # gives the shape of the dataset


# In[15]:


# Do dataset visualizations
print(df.loc[0]) # visualizes the first row
print(df.describe()) #summary of data


# In[16]:


# Plot histogram for each variable
df.hist(figsize = (10, 10))
plt.show()


# In[18]:


# Create scatter plot matrix
scatter_matrix(frame=df, figsize = (18, 18))
plt.show()


# In[19]:


# Create X and Y datasets for training
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


# In[20]:


# Specify testing options
seed = 8 
scoring = 'accuracy'


# In[23]:


# Define the models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC()))
models.append(('DecisionTree', DecisionTreeClassifier(criterion= 'entropy')))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('RandomForest', RandomForestClassifier(n_estimators = 10, 
                                    criterion = 'entropy')))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed) # computes k-fold cross validation
    cv_results = model_selection.cross_val_score(estimator=model, X=X_train, y=y_train, cv=kfold, scoring=scoring) # evaluates a score by cross-validation
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[24]:

cms = []
# Make predictions on validation dataset
for name, model in models:
    model.fit(X=X_train, y=y_train)
    predictions = model.predict(X=X_test)
    print(name)
    print(accuracy_score(y_true=y_test, y_pred=predictions))
    print(classification_report(y_true=y_test, y_pred=predictions))
    #making the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    cms.append(cm)
    print(cm)

# In[ ]:

# Example with new data
for name, clf in models:
    clf.fit(X_train, y_train)
    accuracy = clf.score(X=X_test, y=y_test)
    #print(accuracy)
    example = np.array([[4,2,2,1,10,2,3,2,4], [1,2,2,1,10,2,3,2,10]]) # new data
    #example = example.reshape(len(example), -1)
    prediction = clf.predict(example)
    print("{0}: {1} {2}".format(name, accuracy, prediction))

