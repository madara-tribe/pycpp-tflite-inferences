import sys
sys.path.append('../../')
import os
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import tensorflow as tf
import const as c
from utils import tflite_utils as cfg

batch_size = 128
EPOCHS = 2
epochs = EPOCHS
validation_split = 0.1 # 10% of training set will be used for validation set. 

def create_prune_model(logdir):
    os.makedirs(c.LOGDIR, exist_ok=True)
    
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)}
    model = tf.keras.models.load_model(c.BASE_MODEL_PATH)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model_for_pruning.summary()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_images, train_labels,
                    batch_size=batch_size, epochs=EPOCHS, validation_split=validation_split,
                    callbacks=callbacks)
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)
    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print('Baseline test accuracy:', baseline_model_accuracy) 
    print('Pruned test accuracy:', model_for_pruning_accuracy)
    return model_for_pruning, model_for_pruning_accuracy


def create_more_pruned_model(pruning_model):
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruning_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    quantized_and_pruned_tflite_file = c.MORE_PRUNED_MODEL_PATH 

    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)

    print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)
    print("Size of gzipped baseline Keras model: %.2f bytes" % (cfg.get_gzipped_model_size(c.BASE_MODEL_PATH)))
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (cfg.get_gzipped_model_size(c.MORE_PRUNED_MODEL_PATH)))

