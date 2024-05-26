#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 


# In[2]:


from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2


# In[3]:


encoder = OneHotEncoder()
encoder.fit([[0], [1]]) 

# 0 - Tumor
# 1 - Normal


# In[4]:


path = r'C:\Users\rishi\Downloads\BrainT\brain_tumor_dataset'
print(os.listdir(path))

image_paths = list(paths.list_images(path))
print(len(image_paths))


# In[5]:


for dir in os.listdir(path):
    no_images= len(os.listdir(os.path.join(path,dir)))
    print(no_images)


# In[6]:


# This cell updates result list for images with tumor

data = []
paths = []
result = []

for r, d, f in os.walk(r'C:\Users\rishi\Downloads\BrainT\brain_tumor_dataset\yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())


# In[7]:


# This cell updates result list for images without tumor

paths = []
for r, d, f in os.walk(r'C:\Users\rishi\Downloads\BrainT\brain_tumor_dataset\no'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())


# In[8]:


data = np.array(data)
data.shape


# In[9]:


result = np.array(result)
result = result.reshape(139,2)


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, random_state=0,shuffle=True)


# In[11]:


y_train.shape


# In[80]:


from keras.callbacks import ModelCheckpoint,EarlyStopping

es=EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=1,mode='auto')
mc=ModelCheckpoint(monitor='val_accuracy',filepath='./bestmodel.h5',verbose=1,save_best_only=True,mode='auto')
cd=[es,mc]


# In[73]:


early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)


# In[13]:


history = model.fit(x_train, y_train, epochs = 40, shuffle=True,batch_size = 32,verbose =1,validation_data = (x_test, y_test),callbacks=None)


# In[116]:


history = model.fit(x_train, y_train, epochs = 20, shuffle=True,batch_size = 32,verbose =1,validation_data = (x_test, y_test),callbacks=None)


# In[115]:


history = model.fit(x_train, y_train, epochs = 30, shuffle=True,batch_size = 32,verbose =1,validation_data = (x_test, y_test),callbacks=None)


# In[12]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  #1

model.compile(loss = "categorical_crossentropy", optimizer='Adamax',metrics='accuracy')
print(model.summary())


# In[13]:


# optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam",)
from tensorflow.keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy
optimizer = SGD(learning_rate=0.01)
loss_fn = BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


# In[14]:


history = model.fit(x=x_train, y=y_train, epochs=14, batch_size=20)


# # loss

# In[108]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])   #85 %
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test','Validation'], loc='upper right')
plt.show()


# In[94]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])  #89 %
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# In[40]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# In[109]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])    #85 %
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# In[28]:


#summarize for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])       #82%
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()


# # Accuracy

# In[93]:


#new one
plt.plot(history.history["accuracy"],c="purple")   #89
plt.plot(history.history["val_accuracy"],c="orange")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train","test"])
plt.show()


# In[113]:


#new 
plt.plot(history.history["accuracy"],c="orange")
plt.plot(history.history["val_accuracy"],c="purple")   #85 %
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train","test"])
plt.show()


# In[ ]:





# In[29]:


# summarize history for accuracy old
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')                     #82 %
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[90]:


def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


# In[91]:


from matplotlib.pyplot import imshow
img = Image.open(r"C:\Users\rishi\Downloads\BrainT\brain_tumor_dataset\no\8 no.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)         # This method will show the image in any viewer images
print(str(res[0][classification]*100) + '% Confidence That ' + names(classification))


# In[92]:


from matplotlib.pyplot import imshow
img = Image.open(r"C:\Users\rishi\Downloads\BrainT\brain_tumor_dataset\yes\Y36.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence That ' + names(classification))


# In[59]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)


# In[60]:


print(classification_report(y_test_new,pred))


# In[61]:


cm = confusion_matrix(y_test_new, pred)
print(cm)


# In[87]:


#Final accuracy of the model
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print('Accuracy: {:.4f}%'.format(accuracy))


# In[40]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)


# In[41]:


print(classification_report(y_test_new,pred))


# In[42]:


cm = confusion_matrix(y_test_new, pred)
print(cm)


# In[38]:


import seaborn 
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%',cmap='YlGnBu')  #82%


# In[110]:


import seaborn 
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%',cmap='YlGnBu')  #85%


# In[86]:


import seaborn 
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%',cmap='YlGnBu')


# In[ ]:





# In[ ]:





# In[37]:


#Final accuracy of the model
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print('Accuracy: {:.4f}'.format(accuracy))


# In[89]:


acc=model.evaluate(x_train,y_train)[1]
print('the accuracy of our model is',acc*100,'%')


# In[87]:


acc=model.evaluate(x_test,y_test)[1]
print('the accuracy of our model is',acc*100,'%')


# In[ ]:





# In[112]:


acc=model.evaluate(x_train,y_train)[1]
print('the accuracy of our model is',acc*100,'%')


# In[111]:


acc=model.evaluate(x_test,y_test)[1]
print('the accuracy of our model is',acc*100,'%')


# In[ ]:





# In[27]:


acc=model.evaluate(x_test,y_test)[1]
print('the accuracy of our model is',acc*100,'%')

