from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

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
        img_array= resize(img_array, (64,64,3))
        images.append(img_array)
        img_array = img_array.flatten()
        flat_data.append(img_array)
        target.append(class_num)
           
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
#%%
x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.3)

model=RandomForestClassifier()

model.fit(x_train,y_train)

#%%
y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix , classification_report, precision_score
cm = confusion_matrix(y_test, y_pred)

p = classification_report(y_pred,y_test, target_names=cate)
accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))
print ("Precision = %.3f" % precision_score(y_test, y_pred,average = 'weighted'))


fig, ax = plt.subplots(figsize=(13,15))
sns.heatmap(cm, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False,ax=ax)
