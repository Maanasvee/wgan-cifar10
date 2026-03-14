from model import build_generator, build_critic
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load dataset
(x_train,_),(_,_) = cifar10.load_data()
x_train = (x_train / 127.5) - 1

# Build models
generator = build_generator()
critic = build_critic()

# Training loop
epochs = 10000

for epoch in range(epochs):

    # training code here
    
    if epoch % 1000 == 0:
        print("Epoch:", epoch)

# SAVE GENERATOR MODEL
generator.save("../models/generator.h5")

print("Generator model saved!")
