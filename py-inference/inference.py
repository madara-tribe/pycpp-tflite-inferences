import tensorflow as tf
import numpy as np
import cv2
import os, sys
import time

from utils import tflite_utils as cfg

WEIGHT_DIR = '../weights'
IMG_PATH='36_0_0_face.jpg'
TFLITE_NAME = 'tmp_model.tflite'

def load_data():
    x_test = cv2.imread(IMG_PATH)
    x_test = cv2.resize(x_test, (299, 299))
    x_test = load.to_mean_pixel(x_test, load.MEAN_AVG)
    x_test = x_test.astype(np.float32)/255
    return np.expand_dims(x_test, axis=0)


def tflite_inference(tflite_model_path):
    input_data = load_data()
    print('tflite loading ....')
    interpreter = tf.lite.Interpreter(model_path=os.path.join(WEIGHT_DIR, tflite_model_path))
    print('tflite prediction ...')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    print('start calculation .....')
    start = time.time()
    gender_output = interpreter.get_tensor(output_details[0]['index'])[0]
    age_output = interpreter.get_tensor(output_details[1]['index'])[0]
    gender_output = "M" if np.argmax(gender_output) < 0.5 else "F"
    age_output = np.argmax(age_output)
    print('tflite gender {0} and age {1}'.format(gender_output, int(age_output*100)))
    predict_time = time.time() - start
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")
    
    print('used memory info')
    cfg.show_memory_info()
    
if __name__=='__main__':
    if len(sys.argv)>1:
        tflite_model_path = sys.argv[1]
        tflite_inference(tflite_model_path)
    else:
        print("python3 inference.py <tflite_moel_path>")


