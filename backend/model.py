import tensorflow as tf
from tensorflow.keras import layers

LATENT_DIM = 100

def build_generator():

    model = tf.keras.Sequential([
        layers.Dense(8*8*256, input_dim=LATENT_DIM),
        layers.Reshape((8,8,256)),

        layers.Conv2DTranspose(128,4,strides=2,padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64,4,strides=2,padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(3,3,padding="same",activation="tanh")
    ])

    return model


def build_critic():

    model = tf.keras.Sequential([
        layers.Conv2D(64,3,strides=2,padding="same",input_shape=(32,32,3)),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128,3,strides=2,padding="same"),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1)
    ])

    return model