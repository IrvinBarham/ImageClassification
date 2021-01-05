#Import Necessary Libs
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Import Data from cifar10
from keras.datasets import cifar10
(train_pic, train_labels), (test_pic, test_labels) = cifar10.load_data()

#Classification Arrray
classification = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Normalize Pixels to be Between 0 and 1
train_pic = train_pic / 255
test_pic = test_pic / 255

train_labels_neural = to_categorical(train_labels)
test_labels_neural = to_categorical(test_labels)

#Create Model Architecture
model = Sequential()
#First Layer
model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (32,32,3)))
#Pooling Layer
model.add(MaxPooling2D(pool_size = (2,2)))
#Additional Convulution Layer
model.add(Conv2D(32, (5,5), activation = 'relu'))
#Additional Pooling Layer
model.add(MaxPooling2D(pool_size = (2,2)))
#Flattening Layer
model.add(Flatten())
#Add Layer with 1000 Neurons
model.add(Dense(1000, activation= 'relu'))
#Add Drop Out Layer
model.add(Dropout(0.5))
#Add Layer with 500 Neurons
model.add(Dense(500, activation = 'relu'))
#Add Drop Out Layer
model.add(Dropout(0.5))
#Add Layer with 250 Neurons
model.add(Dense(250, activation = 'relu'))
#Add Layer with 10 Neurons
model.add(Dense(10, activation = 'softmax'))

#Compile the Model
model.compile(loss= 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
#Train Model
hist = model.fit(train_pic, train_labels_neural, batch_size = 256, epochs = 10, validation_split = 0.2)

#Evaluate Model Using Test Data set
model.evaluate(train_pic, train_labels_neural)[1]
                    
#Present Model Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Present Model Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#TESTING TESTING TESTING With Personal Uploaded Images
from google.colab import files
uploaded = files.upload()

#Show Image
new_image = plt.imread('imported_picture.jpg')
img = plt.imshow(new_image)
#Resize
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)

#Retrive Model Predictions
prediction = model.predict(np.array([resized_image]))
#Show
prediction

#Sort Predictions From Least to Greatest
list = [0,1,2,3,4,5,6,7,8,9]
x = prediction
for i in range(10):
  for j in range(10):
    if x[0][list[i]] > x[0][list[j]]:
      temp = list[i]
      list[i] = list[j]
      list[j] = temp

print(list)

#Give 5 best Predictions
for i in range(5):
  print(classification[list[i]], ':', round(prediction[0][list[i]]*100, 2), '%')

