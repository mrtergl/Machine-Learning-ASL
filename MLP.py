import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.optimizers import SGD
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

target = []
images = []
flat_data=[]

ML= 'img//'
PATH = os.path.abspath(os.path.dirname(ML))
cate = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


for category in cate:
    class_num = cate.index(category)
    path = os.path.join(PATH,category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        img_array= resize(img_array, (96,96,1))
        images.append(img_array)
        img_array = img_array.flatten()
        #img_array = img_array.reshape(96,96,1)
        flat_data.append(img_array)
        target.append(class_num)
           
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

#%%
x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.3)
a= x_train[1]
#%%
from keras.models import Sequential, InputLayer
from keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


model=Sequential()

model.add(InputLayer(input_shape=(96*96*1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.add(BatchNormalization())
model.add(Dropout(0.8))

adam = Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#%%
model_history = model.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_test,y_test))
#%%

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
ax = axes.flat

pd.DataFrame(P.history)[['accuracy','val_accuracy']].plot(ax=ax[0])
ax[0].set_title("Accuracy", fontsize = 15)
ax[0].set_ylim(0,1.1)

pd.DataFrame(P.history)[['loss','val_loss']].plot(ax=ax[1])
ax[1].set_title("Loss", fontsize = 15)
plt.show()
type(flat_data[1])
