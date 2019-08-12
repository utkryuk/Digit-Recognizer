# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:06:18 2019

@author: Ezone
"""

import pandas as pd
import numpy as np
from keras.layers.core import Dense,Activation,Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import utils
import matplotlib.pyplot as plt

# Reading and loading data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_test = test.iloc[:,:]


X_train = train.drop(labels =['label'],axis = 1)
Y_train = train.iloc[:, 0]  #the output digit
X_test = X_test/255
X_train = X_train/255

Y_train = utils.to_categorical(Y_train, num_classes = 10)  #to convert the Y_train into categorical variable


#ConvNet requires shape in form(Size,height,width, no. of channels) and currently the shape is in (size,height*width)


X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)  #We can also use .to_numpy() method in place of .values
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

'''
#Checking the values for each label using countplot

import seaborn as sns
cplot = sns.countplot(Y_train)
'''

#Divide training set into train and validation sets 

from sklearn.model_selection import train_test_split
X_train, X_val,y_train, y_val = train_test_split(X_train,Y_train,test_size = 0.2, random_state = 43)


# Layers
model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))

# Callback for reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.00002, verbose=1)
model_checkpoint = ModelCheckpoint(filepath = 'weights.hdf5', verbose = 1, save_best_only = True)

# Adam optimizer
optimizer = Adam(lr=0.0003)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics =['accuracy'])
history = model.fit(X_train, y_train, epochs = 20, batch_size = 32, callbacks = [reduce_lr, model_checkpoint], verbose=2, validation_data = (X_val, y_val))



# Predict classes
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis = 1)

#plotting accuracy with respect to epochs for this model
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
legend_x = 1
legend_y = 1.16
plt.legend(['train', 'test'], loc='upper right',bbox_to_anchor=(legend_x,legend_y))
plt.show()



indices = np.arange(1,predictions.shape[0]+1)
data = {'ImageId': indices, 'Label': predictions}
df = pd.DataFrame(data)
df.to_csv('digits.csv',index = False)