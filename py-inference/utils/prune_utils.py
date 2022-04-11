from utils import tflite_utils as cfg
import tensorflow_model_optimization as tfmot
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

LOGDIR = '/tmp'
WEIGHT_DIR = 'weights'
BASE_MODEL_PATH = 'age_gender_model.hdf5'
PRUNED_MODEL_PATH = 'prune_keras_model.h5'
PRUNED_TFLITE_PATH = 'ep5_pruned_model.tflite'
QUANTIZED_PRUNED_MODEL_PATH = 'quantized_and_pruned_model.tflite'


def cal_model_size():
    print("Size of gzipped baseline Keras model: %.2f bytes" % (cfg.get_gzipped_model_size(os.path.join(WEIGHT_DIR, BASE_MODEL_PATH))))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (cfg.get_gzipped_model_size(os.path.join(LOGDIR, PRUNED_MODEL_PATH))))
    print("Size of gzipped pruned TFlite model: %.2f bytes" % (cfg.get_gzipped_model_size(os.path.join(LOGDIR, PRUNED_TFLITE_PATH))))
    print("Size of gzipped pruned and quantize TFlite model: %.2f bytes" % (cfg.get_gzipped_model_size(os.path.join(LOGDIR, QUANTIZED_PRUNED_MODEL_PATH))))

def save_keras_model(logdir, model_for_pruning):
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    pruned_keras_file = os.path.join(logdir, PRUNED_MODEL_PATH)
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

def save_tflite_model(logdir, model_for_pruning):
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    pruned_tflite_file = os.path.join(logdir, PRUNED_TFLITE_PATH)
    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)
    print("Saved pruned TFLite model to:", pruned_tflite_file)

def evaluate_tflite(interpreter, test_images, gender, age):
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  #print(output_details)
  # Run predictions on ever y image in the "test" dataset.
  gender_prediction = []
  age_prediction = []
  gender_test = []
  age_test = []
  for i, test_image in enumerate(test_images):
    if i % 10 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    age_output = interpreter.get_tensor(output_details[0]['index'])[0]
    gender_output = interpreter.get_tensor(output_details[1]['index'])[0]

    gender_prediction.append(np.argmax(gender_output))
    age_prediction.append(age_output[0])
    gender_test.append(np.argmax(gender[i]))
    age_test.append(age[i])
    if i==100:
        break
  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  gender_prediction = np.array(gender_prediction)
  age_prediction = np.array(age_prediction)
  gender_test, age_test = np.array(gender_test), np.array(age_test)
  
  gender_acc = (gender_prediction == gender_test).mean()
  age_acc = mean_squared_error(age_prediction, age_test)
  return gender_acc, age_acc


