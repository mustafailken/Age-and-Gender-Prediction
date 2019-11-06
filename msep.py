# -*- coding: utf-8 -*-
"""
Created on Mon Dec 8 20:49:29 2018

@author: Mustafa
"""


 
# Keras kütüphanelerinin ve paketlerinin import edilmesi
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# CNN kullanımının başlaması
classifier = Sequential()

#  Kenarları kıvrımları düzenleme
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Birleştirme
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. katmanı ekleme
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Düzleştirme
classifier.add(Flatten())

#  Tam bağlantı
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# CNN ' i compile etme
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.summary()
# CNN 'i görüntülere fit etme

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C://Users/Mustafa/Desktop/Age-Prediction/Gender/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C://Users/Mustafa/Desktop/Age-Prediction/Gender/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history  = classifier.fit_generator(training_set,
                         samples_per_epoch = 200,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 200)

classifier.save("C://Users/Mustafa/Desktop/Age-Prediction/Gender/gclf.h5py")


#Kütüphanelerin eklenmesi.

import keras.models
clf = keras.models.load_model("C://Users/Mustafa/Desktop/Age-Prediction/Gender/gclf.h5py")

import numpy as np
from keras.preprocessing import image
img= image.load_img("",target_size = (64,64))
img= image.img_to_array(img)
round(clf.predict(img[None,:,:,:])[0][0])


import matplotlib.pyplot as plt

x = history.history['loss']
y = history.history['acc']

plt.pie(x,y)




