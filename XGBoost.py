import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import seaborn as sns
from keras.applications.vgg16 import VGG16

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
        img_array= resize(img_array, (96,96,3))
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

x_train, x_test, y_train, y_test = train_test_split(flat_data,target,test_size=0.33,random_state=1)
# VGG16 is a convolution neural net (CNN ) architecture
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3)) 
#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()

feature_extractor=VGG_model.predict(x_train)
#%%
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features #This is our X input to RF
#%%
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_for_training, y_train)

x_test_feat = VGG_model.predict(x_test)
x_test_feat = x_test_feat.reshape(x_test_feat.shape[0], -1)

prediction = model.predict(x_test_feat)
#%%
from sklearn.metrics import confusion_matrix , classification_report
cm = confusion_matrix(y_test, prediction)

fig, ax = plt.subplots(figsize=(13,15))
sns.heatmap(cm, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False,ax=ax)

y_pred_classes = [np.argmax(element) for element in prediction]

ç = classification_report(y_test, prediction, target_names=cate)
from sklearn import metrics
print(classification_report(y_test, prediction, target_names=cate))
print ("Precision = %.3f" % metrics.precision_score(y_test, prediction,average = 'weighted'))

# retrieve performance metrics
results = model.evals_result()
# plot learning curves
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
# show the legend
plt.legend()
# show the plot
plt.show()