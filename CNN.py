import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import confusion_matrix , classification_report
from sklearn import metrics
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization

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
        img_array= resize(img_array, (96, 96,3))
        images.append(img_array)
        img_array = img_array.flatten()
        img_array = img_array.reshape(96,96,3)
        flat_data.append(img_array)
        target.append(class_num)
           
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
#%%
#TRAIN AND TEST

x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.3)


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


def plot_sample(x,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(cate[y[index]])
    
plot_sample(x_train,y_train,1)


#%%.

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same', strides= 2, activation = 'relu', input_shape = (96,96,3)))
model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same', strides= 2, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(1,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', strides = 1, activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same',  strides = 1, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', strides= 1, activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', strides = 1, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(26, activation = "softmax"))
model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#%%
P = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=200,batch_size=40)
#200 epcohs 40 batch size -> 96,96,3- 512Dense, Full cate -> 0.71
#150 epcohs 10 batch size -> 50,50,3- 512Dense, Full cate -> 0.37                       
model.evaluate(x_test,y_test)
#%%
plt.plot(P.history['val_loss'], color='b', label="Test Loss")
plt.plot(P.history['loss'], color='r', label="Train Loss")
plt.title("40BS, Dropout=0.4, Strides 2.2,1.1,1.1,2.2,1.1,2.2")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
#%%
plt.plot(P.history['accuracy'], color='y', label="Train Accuracy")
plt.plot(P.history['val_accuracy'], color='g', label="Test Accuracy")
plt.title("40BS, Dropout=0.4, Strides 2.2,1.1,1.1,2.2,1.1,2.2")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

p = classification_report(y_test, y_pred_classes)
print("Classification Report: \n", classification_report(y_test, y_pred_classes))
print ("Precision = %.3f" % metrics.precision_score(y_test, y_pred_classes,average = 'weighted'))
cm = confusion_matrix(y_test, y_pred_classes)
fig, ax = plt.subplots(figsize=(13,15))
sns.heatmap(cm, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False,ax=ax)
