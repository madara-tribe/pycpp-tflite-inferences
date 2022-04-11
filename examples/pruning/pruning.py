import sys
sys.path.append('../../')
import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import const as c
from utils import tflite_utils as cfg
from create_prune_model import create_prune_model, create_more_pruned_model
#%load_ext tensorboard

def prune_model2tflite(pruning_model):
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruning_model)

    pruned_keras_file = c.PRUNED_MODEL_PATH
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    pruned_tflite_file = c.PRUNED_TFLITE_PATH 
    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)
    print('Saved pruned TFLite model to:', pruned_tflite_file)

def cal_model_size():
    print("Size of gzipped baseline Keras model: %.2f bytes" % (cfg.get_gzipped_model_size(c.BASE_MODEL_PATH)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (cfg.get_gzipped_model_size(c.PRUNED_MODEL_PATH)))
    print("Size of gzipped pruned TFlite model: %.2f bytes" % (cfg.get_gzipped_model_size(c.PRUNED_TFLITE_PATH)))



def evaluate_tflite(interpreter, test_images, test_labels):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on ever y image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_images):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy


def eval_tflite_model(model_for_pruning_accuracy, test_images, test_labels):
    interpreter = tf.lite.Interpreter(model_path=c.MORE_PRUNED_MODEL_PATH)
    interpreter.allocate_tensors()
    test_accuracy = evaluate_tflite(interpreter, test_images, test_labels)
    print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
    print('Pruned TF test accuracy:', model_for_pruning_accuracy)



if __name__=='__main__':
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    more = True
    #jupyter = None
    pruned_model, pruned_model_accuracy = create_prune_model(c.LOGDIR)
    prune_model2tflite(pruned_model)
    cal_model_size()
    if more:
        print('more prune')
        create_more_pruned_model(pruned_model)
        eval_tflite_model(pruned_model_accuracy, test_images, test_labels)
    #if jupyter:
        #%tensorboard --logdir={c.LOGDIR}

