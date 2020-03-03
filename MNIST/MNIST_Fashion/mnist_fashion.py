from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# import helper library
import numpy as np
import matplotlib.pyplot as plt

# import mnist data
# train image 60,000, test image 10,000
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# add labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing the data
plt.figure()
plt.imshow(train_images[0]) # show first image of the training set
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255
test_images = test_images / 255

# print first 25 image 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

# Models

# layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),      # flatten makes 2d array in to 1d array
    keras.layers.Dense(128, activation='relu')
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
predictions = model.predict(test_images)
np.argmax(predictions[0])




