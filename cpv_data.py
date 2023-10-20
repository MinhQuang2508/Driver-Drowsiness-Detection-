#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
import os 
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
 
data_dir = "D:/CPV/data"


# In[51]:


print( os.listdir(data_dir))
train_classes = os.listdir(data_dir + "/train")
print(train_classes)
test_classes = os.listdir(data_dir + "/test")
print(test_classes)


# In[52]:


train_classes = ["Closed", "Open"]  

train_images = 0

for class_name in train_classes:
    class_dir = os.path.join(data_dir, "train", class_name)
    if os.path.exists(class_dir) and os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        num_images = len(images)
        print(f'Total of training examples for {class_name}: {num_images}')
        train_images += num_images
    else:
        print(f"Directory not found for class {class_name}")

print("\nTotal of training images: ", train_images)


# In[53]:


test_classes = ["Closed", "Open"]  

test_images = 0

for class_name in test_classes:
    class_dir = os.path.join(data_dir, "test", class_name)
    if os.path.exists(class_dir) and os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        num_images = len(images)
        print(f'Total of training examples for {class_name}: {num_images}')
        test_images += num_images
    else:
        print(f"Directory not found for class {class_name}")

print("\nTotal of training images: ", test_images)


# In[54]:


def show_images_in_folder(folder_path, num_images=1):
    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            for i in range(min(num_images, len(images))):
                image_path = os.path.join(class_dir, images[i])
                image = plt.imread(image_path)
                plt.figure(figsize=(4, 4))
                plt.imshow(image)
                plt.title(f"Class: {class_name}")
                plt.axis('off')
                plt.show()

num_images_to_show = 1  

show_images_in_folder(os.path.join(data_dir, 'train'), num_images_to_show)


# In[55]:


show_images_in_folder(os.path.join(data_dir, 'test'), num_images_to_show)


# In[56]:


class eye:
  batch_size = 16
  size = 256


# In[57]:


def load_images(directory):
    images = []
    labels = []
    
    for category in os.listdir(directory):
        for filename in tqdm(os.listdir(os.path.join(directory, category))):
            image_path = os.path.join(directory, category, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(category)
    
    images = np.array(images, dtype='float32')
    return images, labels

# Loading dataset
print('*******Loading Train Data*******')
train_ds = load_images(os.path.join(data_dir, 'train'))
print('*******Loading Test Data*******')
test_ds = load_images(os.path.join(data_dir, 'test'))


# In[58]:


x_train,y_train = train_ds
x_test,y_test =  test_ds


# In[59]:


# Calculate the mean and standard deviation from the training data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

# Normalize the training data
x_train_normalized = (x_train - mean) / std

# Normalize the test data using the same mean and standard deviation
x_test_normalized = (x_test - mean) / std


# In[60]:


print(x_train[:5])


# In[61]:


# Label Encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create mapping between original labels and encoded labels
y_train_map = dict(zip(y_train, y_train_encoded))
y_test_map = dict(zip(y_test, y_test_encoded))

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
y_train_encoded = y_train_encoded.reshape(-1, 1)
y_test_encoded = y_test_encoded.reshape(-1, 1)
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded)
y_test_onehot = onehot_encoder.transform(y_test_encoded)


# In[66]:


y_train_map


# In[67]:


y_test_map


# In[ ]:




