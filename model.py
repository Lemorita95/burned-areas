from helpers import *


class Model():

    def __init__(self):

        # Create a convolutional neural network
        model = tf.keras.models.Sequential([

            # input layer
            tf.keras.Input(shape=PNG_RESOLUTION),

            # Convolutional layer 1
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),


            # Convolutional layer 2
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),


            # Convolutional layer 3
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 4
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 5
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 6
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # output layer
            tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding='same'),
        ])

        # Train neural network
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        