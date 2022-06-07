import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import seaborn as sns

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
        img_array= resize(img_array, (100,100,3))
        images.append(img_array)
        img_array = img_array.flatten()
        #img_array = img_array.reshape(50,50,3)
        flat_data.append(img_array)
        target.append(class_num)
           
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
#%%
#TRAIN AND TEST
x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.33,random_state=1)

from sklearn.tree import DecisionTreeClassifier
x_train.shape
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(x_train,y_train)

#%%
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

predictions = clf.predict(x_test)

cm=confusion_matrix(y_test,predictions)
fig, ax = plt.subplots(figsize=(13,15))
sns.heatmap(cm, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False,ax=ax)

print(classification_report(y_test, predictions, target_names=cate))
print ("Precision = %.3f" % metrics.precision_score(y_test, predictions,average = 'weighted'))

