import tensorflow as tf

from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np

model = models.load_model('models/cnn_v1.h5')

print(model)
