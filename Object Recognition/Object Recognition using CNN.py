
# coding: utf-8

# In[6]:


# load the necessary packages
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


# In[24]:


# load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[25]:


# Lets determine the dataset characteristics
print('Training Images: {}'.format(X_train.shape))
print('Testing Images: {}'.format(X_test.shape))


# In[26]:


# A single image
print(X_train[0].shape)


# In[27]:


# create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    img = X_train[i]
    plt.imshow(img)

# show the plot
plt.show()


# In[28]:


# Preprocessing the dataset

# fix a random seed for reproducibility
seed = 6
np.random.seed(seed)

# load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[29]:


print(X_train[0])


# In[21]:


# class labels shape
print(y_train.shape)
print(y_train[0])


# In[30]:


# [6] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] one-hot vector

# hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]

print(num_class)
print(y_train.shape)
print(y_train[0])


# In[36]:


# start by importing necessary layers
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD


# In[50]:


# define the model function

def allcnn(weights):
    
    # defining the model type - Sequential
    model = Sequential()
    
    # add model layers
    model.add(Conv2D(96, (3,3), padding = 'same', input_shape = (32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3,3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3,3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1,1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1,1), padding = 'valid'))
    
    # add Global Average Pooling Layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load the weights
    if weights:
        model.load_weights(weights)
        
    # return the model
    return model


# In[51]:


# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# build the model and define weights
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model summary
print(model.summary())

# define additional training parameters
epochs = 350
batch_size = 32

# test model with pretrained weights
scores = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: {}'.format(scores[1]))

# fit the model
# model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size=batch_size, verbose=1)


# In[52]:


# make a dictionary of class labels and names
classes = range(0, 10)
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# zip the names and classes to make a dictionary of class labels
class_labels = dict(zip(classes, names))
print(class_labels)


# In[ ]:


# generate batch of 9 images to predict
batch = X_test(100:109)
labels = np.argmax(y_test[100:109], axis=1)

# make predictions
predictions = model.predict(batch, verbose = 1)

print(predictions)


# In[ ]:


# these are class probabilities, should sum to 1
for image in predictions:
    print(np.sum(image))


# In[ ]:


# use np.argmax() to convert class probabilities to class labels
class_result = np.argmax(predictions, axis=1)
print(class_result)


# In[ ]:


# create a grid of 3x3 images
fig, axs = plt.subplots(3, 3, figsize=(15,6))
fig.subplots_adjust(hspace=1)
axs=axs.flatten()

for i, img in enumerate(batch):
    
    # determine label for each prediction, set title
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {} \nActual: {}'.format(class_label[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes_get_xaxis().set_visible(False)
            axs[i].axes_get_yaxis().set_visible(False)
        
    # plot the image
    axs[i].imshow(img)

# show the plot
plt.show()

