#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, verbose=0)

# Save the model with .h5 extension
model.save("iris_model.keras")


# In[ ]:




