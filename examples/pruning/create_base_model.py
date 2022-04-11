from tensorflow import keras
import tensorflow as tf
import const as c
batch_size = 128
EPOCHS = 2
epochs = EPOCHS

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


def train():
    # Define the model architecture.
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=EPOCHS, validation_split=0.1)

    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print('Baseline test accuracy:', baseline_model_accuracy)

    keras_file = c.BASE_MODEL_PATH
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)


if __name__=='__main__':
    train()
