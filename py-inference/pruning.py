import os
import warnings
warnings.simplefilter("ignore")

from tensorflow import keras
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import *

from utils import tflite_utils as cfg
import utils.prune_utils as pu
from model_utils import load, util

BATCH_SIZE = 8
EPOCHS = 2
HEIGHT = WIDTH = 299

gender_cls = 2
age_cls=1

TRAIN_DATASET_PATH = '../UTKDATA/UTKFace'
VALID_DATASET_PATH = '../UTKDATA/part3'
WEIGHT_DIR = 'weights'
KERAS_NAME = 'age_gender_model.hdf5'


def model_compile(model, prune=None):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    prune_out = 'prune_low_magnitude_'
    if prune:
        model.compile(optimizer=adam,
                  loss={prune_out+'age_output': 'mae', prune_out+'gender_output': 'binary_crossentropy'},
                  loss_weights={prune_out+'age_output': 0.25, prune_out+'gender_output': 10},
                  metrics={prune_out+'age_output': 'mae', prune_out+'gender_output': 'accuracy'})
    else:
        model.compile(optimizer=adam,
                  loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
                  loss_weights={'age_output': 0.25, 'gender_output': 10},
                  metrics={'age_output': 'mse', 'gender_output': 'accuracy'})
    return model



def create_keras_prune_model(logdir):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) * EPOCHS

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule':
        tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)}
    model = tf.keras.models.load_model(os.path.join(WEIGHT_DIR, KERAS_NAME))
    model = model_compile(model, prune=None)
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # `prune_low_magnitude` requires a recompile.
    model_for_pruning = model_compile(model_for_pruning, prune=True)
    model_for_pruning.summary()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]
    
    print("train for pruning")
    hist1 = model_for_pruning.fit(training, steps_per_epoch=int(len(X_train)/BATCH_SIZE), epochs=EPOCHS,
                       validation_data=(X_val, [age_val, gen_val]),
                       verbose=1, callbacks=callbacks)
    
    print("eval for pruning")
    pruning_acc = model_for_pruning.evaluate(x=X_val, y=[age_val, gen_val], verbose=0)
    baseline_acc = model.evaluate(x=X_val, y=[age_val, gen_val], verbose=0)
    print('model for pruning : age mae:{0}, gender accuracy: {1}'.format(pruning_acc[3], pruning_acc[4]))
    print('base_model: age mae:{0}, gender accuracy: {1}'.format(baseline_acc[3], baseline_acc[4]))
    
    print('saving keras model....')
    pu.save_keras_model(logdir, model_for_pruning)
        
    print('saving tflite model.....')
    pu.save_tflite_model(logdir, model_for_pruning)
        
    return pruning_acc, baseline_acc, X_val, gen_val, age_val








def quantized_and_pruned_tflite_model(X_test, gender_label, age_label, logdir):
    print("create quantized and pruned tflite model")
    pruning_model = tf.keras.models.load_model(os.path.join(logdir, pu.PRUNED_MODEL_PATH))
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruning_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    quantized_and_pruned_tflite_file = os.path.join(logdir, pu.QUANTIZED_PRUNED_MODEL_PATH)
    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
    
    print("quantized pruned model evaluation")
    interpreter = tf.lite.Interpreter(model_path=os.path.join(logdir, pu.QUANTIZED_PRUNED_MODEL_PATH))
    interpreter.allocate_tensors()
    gender_acc, age_acc = pu.evaluate_tflite(interpreter, X_test, gender_label, age_label)
    return gender_acc, age_acc
    
    
    
if __name__=='__main__':
    logdirs = '/tmp'
    pruning_acc, baseline_acc, X_val, gen_val, age_val = create_keras_prune_model(logdir=logdirs)
    gender_acc, age_acc = quantized_and_pruned_tflite_model(X_val, gen_val, age_val, logdir=logdirs)
    
    print('total model accuracy and memory')
    print('keras model for pruning : age mse:{0}, gender accuracy: {1}'.format(pruning_acc[3], pruning_acc[4]))
    print('keras base_model: age mse:{0}, gender accuracy: {1}'.format(baseline_acc[3], baseline_acc[4]))
    print('Pruned and quantized TFLite age mse: {0} and gender accuracy{1}'.format(age_acc, gender_acc))
    pu.cal_model_size()

