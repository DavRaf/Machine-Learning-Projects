
# coding: utf-8

# In[13]:


import sys
import numpy as np
import pandas as pd
import sklearn
import keras

# import the uci pima indians diabetes dataset
filename = "diabetes.csv"
df = pd.read_csv(filename)
df.columns.values


# In[19]:


# Describe the dataset
df.describe()
df[df['Glucose'] == 0]


# In[17]:


# Preprocess the data, mark zero values as NaN and drop
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns:
    df[col].replace(0, np.NaN, inplace=True)
df.describe()


# In[18]:


# Drop rows with missing values
df.dropna(inplace=True)

# Summarize the number of rows and columns in df
df.describe()


# In[21]:


dataset = df.values
print(dataset)
print(dataset.shape)


# In[22]:


# Split into input (X) and output (y)
X = dataset[:, 0:8]
y = dataset[:, 8].astype(int)


# In[24]:


print(X.shape)
print(y.shape)
print(y[:5])


# In[26]:


# Normalize the data using sklearn StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)


# In[27]:


# Transform and display the training data
X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)
data.describe()


# In[30]:


# import necessary packages
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[31]:


# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(4, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())


# In[33]:


# Define a random seed
seed = 6
np.random.seed(seed)
# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(4, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 50, 100]

# make a dictionary of the grid sarch parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(random_state=seed), verbose=10)
grid_result = grid.fit(X_standardized, y)

# summarize the results
print('Best: {0}, using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[34]:


# import necessary packages
from keras.layers import Dropout

# Define a random seed
seed = 6
np.random.seed(seed)
# Start defining the model
def create_model():
    # create model
    model = Sequential(learn_rate, dropout_rate)
    model.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 20, verbose = 0)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2]

# make a dictionary of the grid search parameters
param_grid = dict(learn_rate = learn_rate, dropout_rate=dropout_rate)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(random_state=seed), verbose=1)
grid_result = grid.fit(X_standardized, y)

# summarize the results
print('Best: {0}, using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[36]:


# Define a random seed
seed = 6
np.random.seed(seed)
# Start defining the model
def create_model(activation, init):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer = init , activation = activation))
    model.add(Dense(4, input_dim = 8, kernel_initializer = init, activation = activation))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.1)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 20, verbose = 0)

# define the grid search parameters
activation = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# make a dictionary of the grid search parameters
param_grid = dict(activation = activation, init = init)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(random_state=seed), verbose=1)
grid_result = grid.fit(X_standardized, y)

# summarize the results
print('Best: {0}, using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[37]:


# Define a random seed
seed = 6
np.random.seed(seed)
# Start defining the model
def create_model(neuron1, neuron2):
    # create model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 8, kernel_initializer = 'uniform' , activation = 'tanh'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer = 'uniform', activation = 'tanh'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.1)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 20, verbose = 0)

# define the grid search parameters
neuron1 = [4, 8, 16]
neuron2 = [2, 4, 8]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(random_state=seed), refit = True, verbose=10)
grid_result = grid.fit(X_standardized, y)

# summarize the results
print('Best: {0}, using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[44]:


# Define a random seed
seed = 6
np.random.seed(seed)
# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim = 8, kernel_initializer = 'uniform' , activation = 'tanh'))
    model.add(Dense(2, input_dim = 4, kernel_initializer = 'uniform', activation = 'tanh'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.1)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 20, verbose = 0)


# In[45]:


# generate predictions with optimal hyperparameters
y_pred = grid.predict(X_standardized)

print(y_pred.shape)

print(y_pred[:5])


# In[53]:


# generate a classification report
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))


# In[51]:


# example datapoint
example = df.iloc[5]
print(example)


# In[52]:


# make a prediction using our optimized deep neural network
prediction = grid.predict(X_standardized[5].reshape(1, -1))
print(prediction)

