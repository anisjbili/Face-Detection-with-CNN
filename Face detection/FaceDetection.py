# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:27:02 2018

@author: ASUS
"""

from keras.models import Sequential
import matplotlib.image as img

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import numpy as np
from keras.layers import Flatten
from keras.layers import Dense
import glob
from sklearn.model_selection import train_test_split 

num_classes = 15

files = glob.glob ("faces/*.pgm")
X_data=[]
labelstr = []
for myFile in files:
    
    image= img.imread(myFile)
   # result=np.array(image).flatten().reshape(image.shape[1]*image.shape[0])
    X_data.append (image)
    labelstr.append((myFile.split(".")[0]).split("t")[1])

label = list(map(int, labelstr))
    
X_data = np.array(X_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, label, test_size = 0.3)
X_train = X_train.reshape(-1,243,320,1)
X_test = X_test.reshape(-1,243,320,1)

Model = Sequential()
Model.add(Convolution2D(32, kernel_size=(3, 3), strides=(1,1),
                 activation='relu', input_shape=(243, 320, 1)))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())

Model.add(Dense(100,input_dim=320,activation='relu'))
Model.add(Dense(30,activation='relu'))
Model.add(Dense(15, activation='softmax'))
Model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

Model.fit(X_train, Y_train, epochs=5, batch_size=20)
evaluation = Model.evaluate(X_test, Y_test)
print("accuracy", evaluation[1])