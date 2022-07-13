import cv2
from PIL import Image
import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

"""
Implement the testing procedure here. 

Inputs:
    After unzipping data.zip, run prediction.py with the two data sets in the same directory. 
Outputs:
    A file named "prediction.txt":
        * The prediction file will have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
"""
x_train = []
y_train = []
directory = './train_set'
classes = ['T_shirt/Tops', 'Trousers', 'Pullovers', 'Dresses', 'Coats', 'Sandals', 'Shirts', 
            'Sneakers', 'Bags', 'Ankle Boots']

x_train=[]
y_train = []
# loop 0-9 through the training directories
# associate directory name with y-train label for each picture arrays
for i in range(10):
    curPath = directory+"/"+str(i)
    for pic in os.listdir(curPath):
        if pic.endswith(".png"):
            temp = Image.open(curPath+"/"+pic)
            tempNParr = np.asarray(temp)/255.0
            x_train.append(tempNParr)
            y_train.append(i)
            # sanity check
            if(len(y_train)!=len(x_train)):
                print("ERROR") 
    print("Finished directory " + str(i) + ": " +classes[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

# populate x_test with testing data
x_test = []
curPath = './test_set'
# test loop must be hardcoded to image names, because order of images in directory matters
for i in range(len(os.listdir(curPath))-1):
    picPath = curPath + "/" + str(i) + ".png"
    testpic = Image.open(picPath)
    testNParr = np.asarray(testpic)
    x_test.append(testNParr)
test = np.array(x_test)
test = test/255.0

# for consistency in rng
tf.random.set_seed(10)

# cnn model, conv->pool->conv->pool->conv, flatten->64dense->10dense
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.2),
    Flatten(), 
    Dense(64,activation='relu'),
    Dense(10,activation = 'softmax')
])
# compiling with crossentropy loss
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

print("\nModel trained, predicting now.\n")

model.fit(x_train, y_train, epochs=25)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test)
with open('prediction.txt', 'w') as w:
        for i in range(10000):
            w.write(str(np.argmax(predictions[i])))
            w.write('\n')