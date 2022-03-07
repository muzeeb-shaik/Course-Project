# Link : https://www.learnopencv.com/image-classification-using-feedforward-neural-network-in-keras/
######################## CSCE 633 Project Feed Forward Neural Net #############################


from platform import python_version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics 
import statsmodels.stats.api as sms
import math
import os
import sklearn
from sklearn import tree
from sklearn.tree import export_text
print(python_version())
eps = np.finfo(float).eps
#Python Version 3.7.4

# Change the working directory
os.chdir('D:\\Muzeeb\\ML Class\\Project\\CSCE633_S20_Ananya_Gargi_Keerat_Muzeeb_Project')

cwd = os.getcwd()
print(cwd)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)

#Loading Data
data_filename = "image_features.txt"
image_data  =pd.read_csv(data_filename,header = None)
new_header = image_data.iloc[0]
image_data = image_data[1:]
image_data.columns = new_header 
for i in new_header:
    image_data[i] = image_data[i].astype(float)
image_data['label'] =image_data['label'].replace(-1,0)
Y_list = ['label']
## If we want to run on Specific data as X, we can make a list of those variables as X_list
X_list = list(image_data.columns)
X_list.remove('label')
    
# Randomly shuffling the data
image_data = image_data.sample(frac=1, random_state=30).reset_index(drop=True)
np.random.seed(0)
cv_split = np.array_split(image_data,3)  

split_1 = cv_split[0]
split_2 = cv_split[1]
split_3 = cv_split[2]

train_1 = split_1.append(split_2)
X_train_1 = np.array(train_1[X_list])
y_train_1 = np.array(train_1[Y_list])
test_1  = split_3
X_test_1 = np.array(test_1[X_list])
y_test_1 = np.array(test_1[Y_list])

train_2 = split_1.append(split_3)
X_train_2 = np.array(train_2[X_list])
y_train_2 = np.array(train_2[Y_list])
test_2  = split_2
X_test_2 = np.array(test_2[X_list])
y_test_2 = np.array(test_2[Y_list])

train_3 = split_3.append(split_2)
X_train_3 = np.array(train_3[X_list])
y_train_3 = np.array(train_3[Y_list])
test_3  = split_1
X_test_3 = np.array(test_3[X_list])
y_test_3 = np.array(test_3[Y_list])

#### Perceptron
from sklearn.linear_model import Perceptron
clf1 = Perceptron(tol=1e-3, random_state=0)
clf1.fit(X_train_1, y_train_1)
clf1.score(X_test_1,y_test_1 )
pred_test_1 = clf1.predict(X_test_1)
 
 # confusion matrix
conf_matrix_test1 =  perf_measure(y_test_1,pred_test_1)

clf2 = Perceptron(tol=1e-3, random_state=0)
clf2.fit(X_train_2, y_train_2)
clf2.score(X_test_2,y_test_2)
pred_test_2 = clf2.predict(X_test_2)
 
 # confusion matrix
conf_matrix_test2=  perf_measure(y_test_2,pred_test_2)



############################# FFNN ##################
from sklearn.neural_network import MLPClassifier
clf_ffnn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(300, 2), random_state=1, activation =  'logistic')

clf_ffnn.fit(X_train_1, y_train_1)
clf_ffnn.score(X_test_1, y_test_1)
pred_test_1 = clf_ffnn.predict(X_test_1)
 # confusion matrix
conf_matrix_test1 =  perf_measure(y_test_1,pred_test_1)

############ Kres FNN
#Evaluation result on Test Data : Loss = 0.5468575768745862, accuracy = 0.9384615421295166
#Evaluation result on Test Data : Loss = 0.20668215111854457, accuracy = 0.9634615182876587
#Evaluation result on Test Data : Loss = 0.22759713536271683, accuracy = 0.9173076748847961

import numpy as np
import matplotlib.pyplot as plt

# Find the unique numbers from the train labels
classes = np.unique(image_data['label'])
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
dimData =(image_data.shape[1])-1

train_data =X_train_1
test_data = X_test_1

train_labels = y_train_1
test_labels = y_test_1

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')


# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

train_labels = [tuple(x) for x in train_labels]
test_labels= [tuple(x) for x in test_labels]
# Change the labels from integer to categorical data
from keras.utils import to_categorical

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[21])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[21])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(dimData,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
epoch_list = []
batch_list =[]
test_loss_list =[]
test_acc_list = []
for epoch in range(1,20,5):
    for batch in range (1,len(train_data), 50):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_data, train_labels_one_hot, batch_size=batch, epochs=epoch, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))

        [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
        epoch_list.append(epoch)
        batch_list.append(batch)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)


print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

acc_data =pd.DataFrame()
acc_data['epoch'] =epoch_list
acc_data['batch'] =batch_list
acc_data['test_loss'] =test_loss_list
acc_data['test_acc'] =test_acc_list
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
threedee = plt.figure().gca(projection='3d')
threedee.scatter(acc_data['epoch'], acc_data['batch'], acc_data['test_acc'])
threedee.set_xlabel('Epochs')
threedee.set_ylabel('Batch Size')
threedee.set_zlabel('Accuracy')
plt.show()

### Check for over fitting
#
##Plot the Loss Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['loss'],'r',linewidth=3.0)
#plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)
#
##Plot the Accuracy Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['accuracy'],'r',linewidth=3.0)
#plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves',fontsize=16)
#
#from keras.layers import Dropout
#
#model_reg = Sequential()
#model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
#model_reg.add(Dropout(0.5))
#model_reg.add(Dense(512, activation='relu'))
#model_reg.add(Dropout(0.5))
#model_reg.add(Dense(nClasses, activation='softmax'))
#
#model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1, 
#                            validation_data=(test_data, test_labels_one_hot))
#
##Plot the Loss Curves
#plt.figure(figsize=[8,6])
#plt.plot(history_reg.history['loss'],'r',linewidth=3.0)
#plt.plot(history_reg.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)
#
##Plot the Accuracy Curves
#plt.figure(figsize=[8,6])
#plt.plot(history_reg.history['accuracy'],'r',linewidth=3.0)
#plt.plot(history_reg.history['val_accuracy'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves',fontsize=16)
