from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.utils
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras import datasets, layers, models, Sequential

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize dataset 
train_images = train_images / 255
test_images =  test_images / 255


model = models.Sequential()
#Convolution
model.add(keras.layers.Conv2D(filters = 64, kernel_size =[3,3], padding = 'same', activation = 'relu', input_shape = (32 ,32, 3)))
#Pooling
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#Convolution 2
model.add(keras.layers.Conv2D(filters = 64, kernel_size =[3,3], padding = 'same', activation = 'relu'))
#Pooling 2 
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


#Fully Connected Layer

#Flatten data into 1D
model.add(keras.layers.Flatten())


model.add(keras.layers.Dropout(.2))
#Hidden layers of 128 nodes
model.add(keras.layers.Dense(128, activation = 'relu'))

model.add(keras.layers.Dropout(.2))

#Output Layer
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'] )
#Output
model.fit(train_images, train_labels, batch_size =32, epochs = 8, validation_data=(test_images, test_labels))

