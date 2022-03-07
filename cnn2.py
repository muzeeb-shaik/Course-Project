from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import cv2
#import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras as k
from sklearn.metrics import confusion_matrix, f1_score
#from keras.metrics import accuracy
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import random
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)


img_set  = []
currpath = os.getcwd()
dir1 = "Covid64"
svpath1 = os.path.join(currpath, dir1)
filedata = os.listdir(svpath1)
n = 0
for f in filedata:
  i_path = os.path.join(svpath1, f)
  img = cv2.imread(i_path)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_img = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img_set.append(gray_img)

  #if(n == 15):
  #  break
  n = n + 1

len1 = len(img_set)
label1 = np.zeros(len(img_set))
dir1 = "Normal64"
svpath1 = os.path.join(currpath, dir1)
filedata = os.listdir(svpath1)
n = 0
for f in filedata:
  i_path = os.path.join(svpath1, f)
  img = cv2.imread(i_path)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_img = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img_set.append(gray_img)
  #if(n == 15):
  #  break
  n = n + 1
#label2 = np.ones(len(img_set) - len1) * (-1)
label2 = np.ones(len(img_set) - len1)

img_label = np.append(label1,label2)
print(img_label)
print(len(img_label))
print(img_set)
#img_set = cv2.normalize(img_set, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#img_set = img_set/255
#print(img_set)

img_label = np.array(img_label)
img_label = to_categorical(img_label)
print(img_label)
#StratifiedKFold(n_splits=1, random_state=None, shuffle=False)
#train_index, test_index =  skf.split(img_set, img_label)
train_images, test_images, train_labels, test_labels = train_test_split(img_set, img_label, test_size=0.25, random_state=30,stratify=img_label)
#print(test_labels)
print("************************")
#print(to_categorical(test_labels))
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 16
filter_size = 7
pool_size = 2
bt_size = 128
optimizr = 'adam'
lossf = 'categorical_crossentropy'


# Build the model.
model = Sequential()
#model = Sequential([Conv2D(num_filters, filter_size, input_shape=(64, 64, 1)),MaxPooling2D(pool_size=pool_size),
#  Flatten(), Dense(3, activation='softmax'),])
model.add(Conv2D(num_filters, filter_size,activation='relu', input_shape=(64, 64, 1)),)
model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
# Compile the model.
model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)

# Train the model.
history = model.fit(train_images,train_labels,batch_size=bt_size,validation_split=0.20, epochs=20,shuffle = False)

# Save the model to disk.
model.save_weights('cnn_2class.h5')

predictions = model.predict(test_images)
print("predictions")



print("--------------------------------------")
acc1 = k.metrics.accuracy(test_labels, predictions)

acc = np.mean(np.equal(test_labels, np.round(predictions)))
#print("Test set accuracy = ",acc)

[test_loss, test_acc] = model.evaluate(test_images,test_labels)
print("Stats for 20 runs")
print("Accuracy for batch size ",bt_size," = ",test_acc)
pred = np.argmax(predictions, axis=1)
#print("Predicted value = ",pred)
#print("length = ",len(pred))
actual = np.argmax(test_labels, axis=1)
#print(type(pred),len(pred))
#print(type(actual),len(actual))
val_f1 = f1_score(actual, pred,labels=[0])
print("F1 score for batch size ",bt_size," = ",val_f1)
print("\n\n**************************************************\n")

df = pd.DataFrame(pred,columns=['Predicted values'])
df.to_csv('predicted_values_cnn.txt', index=False)

cnt = 0
for i in range(len(test_labels)):
  if (np.array_equal(test_labels[i],np.round(predictions[i]))):
    cnt+=1
#print("accuracy1 = ",cnt/len(test_labels))

testv = np.equal(test_labels, np.round(predictions))

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs validation 2 class')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Acc', 'Val Acc'], loc='lower right')
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'CNN_output')
#print("curr_dir = ", current_directory)
if not os.path.exists(final_directory):
  os.makedirs(final_directory)
figpath = "./CNN_output/"
name = figpath + "acc_graph_2class"  + '.png'
plt.savefig(name)
acc_vect = []
#plt.show()

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history1 = model.fit(train_images,train_labels,batch_size=1,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
acc_vect.append(test_acc)
print("Accuracy for batch size 1 = ",test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_1 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history1 = model.fit(train_images,train_labels,batch_size=4,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
acc_vect.append(test_acc)
print("Accuracy for batch size 4 = ",test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_4 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history2 = model.fit(train_images,train_labels,batch_size=16,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
acc_vect.append(test_acc)
print("Accuracy for batch size 16 = ",test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_16 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history3 = model.fit(train_images,train_labels,batch_size=32,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
acc_vect.append(test_acc)
print("Accuracy for batch size 32 = ",test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_32 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history3 = model.fit(train_images,train_labels,batch_size=64,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
acc_vect.append(test_acc)
print("Accuracy for batch size 64 = ",test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_64 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history4 = model.fit(train_images,train_labels,batch_size=256,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
print("Accuracy for batch size 256 = ",test_acc)
acc_vect.append(test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_256 = f1_score(actual, pred,labels=[0])

model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history4 = model.fit(train_images,train_labels,batch_size=512,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
print("Accuracy for batch size 512 = ",test_acc)
acc_vect.append(test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_512 = f1_score(actual, pred,labels=[0])

len2 = len(train_images)
model.compile(optimizer=optimizr,loss=lossf,metrics=['accuracy'],)
history4 = model.fit(train_images,train_labels,batch_size=len2,validation_split=0.20, epochs=6,verbose=0)
[test_loss, test_acc] = model.evaluate(test_images,test_labels,verbose=0)
print("Accuracy for batch size all (",len2, ") =",test_acc)
acc_vect.append(test_acc)
predictions = model.predict(test_images)
pred = np.argmax(predictions, axis=1)
val_f1_len = f1_score(actual, pred,labels=[0])

print("F1 score for batch size 1 = ",val_f1_1)
print("F1 score for batch size 4 = ",val_f1_4)
print("F1 score for batch size 16 = ",val_f1_16)
print("F1 score for batch size 32 = ",val_f1_32)
print("F1 score for batch size 64 = ",val_f1_64)
print("F1 score for batch size 256 = ",val_f1_256)
print("F1 score for batch size 512 = ",val_f1_512)
print("F1 score for batch size all (",len2, ") =",val_f1_len)

print(acc_vect)
#acc_vect = [round(x,5) for x in acc_vect]
plt.figure()
plt.title('Accuracy vs BatchSize for 2 class')
plt.ylabel('Accuracy')
plt.xlabel('Batch Size')
epoch = ['1','4','16','32','64','256','512',str(len2)]
#acc_vect = [98.1,98.1,98.3,98.2,98.1,98.4,]
#plt.xlim([0, 600])
plt.ylim([0.95, 1])
#plt.legend(['Train Acc', 'Val Acc'], loc='lower right')
#plt.plot(epoch,acc_vect,color='green', linestyle='dashed', linewidth = 3)
#plt.bar(epoch, acc_vect,width = 0.2,tick_label = epoch)
plt.scatter(epoch, acc_vect, label= "stars", color= "green", marker= "*", s=30)
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'CNN_output')
#print("curr_dir = ", current_directory)
if not os.path.exists(final_directory):
  os.makedirs(final_directory)
figpath = "./CNN_output/"
name = figpath + "acc_batchSize_2class"  + '.png'
plt.savefig(name)
