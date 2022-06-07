import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.optimizers import SGD


target = []
images = []
flat_data=[]

ML= 'img//'
PATH = os.path.abspath(os.path.dirname(ML))
cate = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']



for category in cate:
    class_num = cate.index(category)
    path = os.path.join(PATH,category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        img_array= resize(img_array, (75,75,3))
        images.append(img_array)
        img_array = img_array.flatten()
        img_array = img_array.reshape(75,75,3)
        flat_data.append(img_array)
        target.append(class_num)
           
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.3)
#%%
# Loading inception v3 network for transfer learning
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
WEIGHTS_FILE = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

inception_v3_model = InceptionV3(
    input_shape = (75, 75, 3), 
    include_top = False, 
    weights = 'imagenet'
)
model = Model(inception_v3_model.input, x) 

#%%


from tensorflow.keras.optimizers import RMSprop, Adam, SGD

x = layers.GlobalAveragePooling2D()(inception_output)
x = layers.Dense(1024, activation='relu')(x)
# Not required --> x = layers.Dropout(0.2)(x)                  
x = layers.Dense(29, activation='softmax')(x)           

model = Model(inception_v3_model.input, x) 
model.summary()


#%%
model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['acc']
)
model.fit(x_train, y_train, epochs=50)






