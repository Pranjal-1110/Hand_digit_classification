# MNIST Handwritten Digit Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The CNN is built using TensorFlow and Keras and achieves around 98% accuracy on the validation data.

## Table of Contents
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Dependencies

The following dependencies are required to run this project:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers 
```
## Dataset
The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset is divided into two parts: 60,000 training images and 10,000 testing images.

## Preprocessing
### Image Normalization
The images are normalized to the range [0, 1] to facilitate faster convergence during training.
```python
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
```
### Resizing the Images
The images are reshaped to include an additional dimension required for the CNN.
```python
img_dim = 28
x_train = np.array(x_train).reshape(-1, img_dim, img_dim, 1)
x_test = np.array(x_test).reshape(-1, img_dim, img_dim, 1)
```
### One-Hot Encoding Labels
The labels are one-hot encoded to match the output of the softmax layer in the CNN.
```python
y_trainr = to_categorical(y_train, 10)
y_testr = to_categorical(y_test, 10)
```
## Model Architecture
The CNN model consists of the following layers:
<li>Two convolutional layers with ReLU activation and max-pooling layers.</li>
<li>A flatten layer to convert the 2D matrix to a 1D vector.</li>
<li>Two fully connected dense layers with ReLU activation and a dropout layer to prevent overfitting.</li>
<li>An output dense layer with softmax activation.</li>

```python
model = Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])

```
## Training
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. It is trained for 10 epochs with a batch size of 200.
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_trainr, validation_data=(x_test, y_testr), epochs=10, batch_size=200, verbose=2)
```
## Evaluation
The model's performance is evaluated by plotting the training and validation accuracy and loss over the epochs.
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

```
## Results
The model achieved an accuracy of approximately 98% on the validation data. The prediction results are visualized by comparing the true labels and predicted labels for the first 100 test samples.
```python
def predict(n):
    final_predict = []
    for i in range(n):
        final_predict.append(np.argmax(predictions[i]))
    return final_predict
        
plt.scatter(x=np.arange(1, 101), y=y_test[0:100])
plt.scatter(x=np.arange(1, 101), y=predict(100))
```
