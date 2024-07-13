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
