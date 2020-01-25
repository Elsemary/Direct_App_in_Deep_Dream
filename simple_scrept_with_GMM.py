# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:23:11 2020

@author: ABDELRHMAN H
"""
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
import skimage
import pandas as pd
from sklearn import svm, metrics
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

list_alpha=['A.png','B.png','C.png','D.png','E.png','F.png']
list_num=['1','2','3','4','5']

full_image=[]
for i in range(len(list_alpha)):
    for j in list_num:
        img=mpimg.imread(j+list_alpha[i])
        full_image.append(skimage.util.img_as_uint(img))

iamgee=[]
for f in range (len(full_image)):
    for i in range (len(full_image[f])):
        o=pd.DataFrame(full_image[f][i,:,:])[0]
        p=np.array(o).reshape(1,-1)
        iamgee.append(p)
kk=[0]
for k in range (len(full_image)):
        l=len(full_image[k])
        kk.append(l)

kk1=np.cumsum(kk)

images=[]
for k in range (len(full_image)):
    kk=iamgee[kk1[k]:kk1[k+1]]
    _img=np.row_stack(kk)
    images.append(_img)

y=[]
for h_ in range (len(images)):
    for i in range (images[h_].shape[0]):
        xn=images[h_][i]
        y.append(xn)

s__=[]
features=[]
for s_ in range (len(full_image)):
    xs=np.concatenate(y[kk1[s_]:kk1[s_+1]]).reshape(1,-1)
    if len(xs[0]) == len(y[kk1[0]:kk1[1]])*len(y[len(y)-1]):
        features.append(s_)
        s__.append(xs)
#plt.imshow(images[1], cmap='gray')

data_=np.asarray(s__)
data=np.concatenate(data_)
target=np.array([0,0,0,1,1,
                 1,2,2,2,3,3,3,4,4,4,5,5,5])

images_and_labels = list(zip(data_, target))


_, axes = plt.subplots(2, 4)
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)



n_samples = len(data_)
# Create a model: a GMM
model = GaussianMixture(len(target))

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.5, shuffle=True)


# We learn the emotion on the first half of the digits
model.fit(X_train, y_train)
predicted = model.predict(X_test)
images_and_predictions = list(zip(data_[n_samples // 2:], predicted))


for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    plt.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title('Prediction: %i' % prediction)

print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_test, predicted)))

plt.show()
