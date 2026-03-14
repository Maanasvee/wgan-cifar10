from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train,_),(_,_) = cifar10.load_data()

x_train = (x_train / 127.5) - 1