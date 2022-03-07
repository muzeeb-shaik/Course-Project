import numpy as np
import pandas as pd
import keras
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

currdir = os.path.dirname(os.path.realpath('__file__'))
print(currdir)
train_path = currdir + "\\COVID-19 Radiography Database\\Train"
validation_path = currdir + "\\COVID-19 Radiography Database\\Validation"
test_path = currdir + "\\COVID-19 Radiography Database\\Test"

batch_size = 64

datagen = ImageDataGenerator(rescale=1. / 255)
train_data = datagen.flow_from_directory(train_path,target_size=(64, 64), color_mode='grayscale', batch_size=batch_size, class_mode="binary",shuffle=False)
validation_data = datagen.flow_from_directory(validation_path,target_size=(64, 64), color_mode='grayscale', batch_size=1, class_mode="binary",shuffle=False)
test_data = datagen.flow_from_directory(test_path,target_size=(64, 64), color_mode='grayscale', batch_size=1, class_mode=None,shuffle=False)

"""
image_data = pd.read_csv("image_features_64.txt")
data = image_data.iloc[:,0:-1]
labels = image_data.iloc[:,-1]
"""

model = Sequential()
model.add(Dense(512, input_shape=(64, 64, 1), init = 'normal', activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.fit(train_data, epochs=10, verbose=1)
history = model.fit_generator(train_data, steps_per_epoch = 2000 // batch_size, epochs = 20, validation_data=validation_data, validation_steps=800 // batch_size, shuffle=False, verbose=1)
#prediction = model.predict_genarator(test_data)
#print(test_data)
#print(prediction)
model.save('webapp_for_diagnosis/models/ffnn_model.h5')


#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.show()


