import sys
sys.path.append('../../')
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import model_selection
import numpy as np
from tqdm import tqdm
from utils import tflite_utils as cfg

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape([60000, 28, 28, 1])
x_test = x_test.reshape([10000, 28, 28, 1])

x_test, x_val, y_test, y_val = model_selection.train_test_split(
        x_test, y_test, test_size=0.1, stratify=y_test)

def keras2tflite():
    print('converting......')
    tflite_model = tf.keras.models.load_model('digit_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    tflite_save = converter.convert()
    open("digit_model.tflite", "wb").write(tflite_save)

def cal_acc():
    print('keras load')
    model = tf.keras.models.load_model('digit_model.h5')
    y_pred1 = model.predict(x_test)
    y_pred1 = np.argmax(y_pred1, axis=1)
    print('keras_model accracy', sum(y_pred1 == y_test) / len(y_test))

    print('tflite load')
    interpreter = tf.lite.Interpreter(model_path="digit_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
        #=> [{'name': 'conv2d_6_input', 'index': 7, 'shape': array([ 1, 28, 28,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
    output_details = interpreter.get_output_details()
        #=> [{'name': 'dense_11/Softmax', 'index': 16, 'shape': array([ 1, 10], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]

    def predict(i):
        input_data = x_test.astype(np.float32)[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    y_pred2 = np.array([predict(i) for i in tqdm(range(len(y_test)))])
    y_pred3 = np.argmax(y_pred2, axis=1)
    print('tflite accuracy', sum(y_pred3 == y_test) / len(y_test))
    cfg.show_memory_info()

def main():
    keras2tflite()
    cal_acc()

if __name__=='__main__':
    main()
