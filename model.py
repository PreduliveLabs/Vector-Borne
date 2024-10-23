import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import joblib


train=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("regions",
                                    batch_size=3,
                                    target_size=(250,250))
validation_data=train.flow_from_directory("regions",
                                    batch_size=3,
                                    target_size=(250,250))

model = tf.keras.models.Sequential()

# Convolutional & Max Pooling layers
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(250,250,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten & Dense layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))

# performing binary classification
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy']
             )
model.fit(train_data,validation_data=validation_data,epochs=10)


joblib.dump(model,"CNN")
