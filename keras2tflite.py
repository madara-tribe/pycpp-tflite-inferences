import tensorflow as tf
import os

WEIGHT_DIR = 'weights'
KERAS_WEIGHT='ep1age_gender_299x299.hdf5'
TFLITE_NAME = 'age_gender_model.tflite'

def keras2tflite(path):
    print('converting......')
    tflite_model = tf.keras.models.load_model(os.path.join(WEIGHT_DIR, KERAS_WEIGHT))
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    tflite_save = converter.convert()
    open(os.path.join(path, TFLITE_NAME), "wb").write(tflite_save)
    
if __name__=='__main__':
    keras2tflite(path=WEIGHT_DIR)

