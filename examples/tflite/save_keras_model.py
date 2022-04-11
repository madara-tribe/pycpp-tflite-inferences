import sys
sys.path.append('../../')
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import model_selection
import os
from utils import tflite_utils as cfg
EPOCHS = 1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape([60000, 28, 28, 1])
x_test = x_test.reshape([10000, 28, 28, 1])

x_test, x_val, y_test, y_val = model_selection.train_test_split(
        x_test, y_test, test_size=0.1, stratify=y_test)

        
def train():
    model = tf.keras.models.Sequential([
        layers.Convolution2D(64, (4, 4), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        layers.Convolution2D(64, (4, 4), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # tensoboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    # fit
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), callbacks=[tensorboard_callback])
    model.save('digit_model.h5')
    cfg.show_memory_info()

if __name__=='__main__':
    train()
