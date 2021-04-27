from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


(trainX, trainY), (testX, testY) = cifar10.load_data()
# one hot encode target values
trainY_bin_class = to_categorical(trainY)
testY_bin_class = to_categorical(testY)

pass