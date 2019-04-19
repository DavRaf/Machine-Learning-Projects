
# coding: utf-8

# In[1]:


# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[14]:


# load the dataset
digits = load_digits()


# In[15]:


# split the data into train and test set
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape


# In[16]:


# normalize data
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


# In[18]:


# create a K-Nearest Neighbors model for each different value of k
from sklearn.neighbors import KNeighborsClassifier

ks = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20]

# k=3 gives the best results
for k in ks:
    print("K="+str(k))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred_train = knn.predict(X_train)
    y_prob_train = knn.predict_proba(X_train)
    
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred)
    
    loss_train = log_loss(y_train, y_prob_train)
    loss_test = log_loss(y_test, y_prob)
    
    print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
    print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train, loss_test))


# In[27]:


# check where the model has predicted bad values
for i in range(0, len(X_test)):
    if(y_test[i]!=y_pred[i]):
        print("Number %d classified as %d" % (y_test[i], y_pred[i]))
        plt.imshow(X_test[i].reshape([8, 8]), cmap="gray")
        plt.show()

