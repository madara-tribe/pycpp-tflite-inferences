# https://docs.w3cub.com/tensorflow~2.3/io/write_graph
import tensorflow as tf
import os
from tensorflow.compat.v1.keras import backend as K

WEIGHTS = 'weights'
KERAS_NAME = 'ep1age_gender_299x299.hdf5'
CKPT_NAME = 'infer_graph.pb'

keras_model = tf.keras.models.load_model(os.path.join(WEIGHTS, KERAS_NAME))
def output_node(keras_model):
    output_names=[out.op.name for out in keras_model.outputs]
    #saver = tf.compat.v1.train.Saver
    print ("\n TF input node name:")
    print(keras_model.inputs)
    print ("\n TF output node name:")
    print(keras_model.outputs)
    
def keras2tfpb():
    # fetch the tensorflow session using the Keras backend
    sess = K.get_session()
    # get the tensorflow session graph
    graph_def = sess.graph.as_graph_def()
    tf.io.write_graph(graph_def, logdir=WEIGHTS, name=CKPT_NAME, as_text=False)
    print("\nFINISHED CREATING TF FILES\n")
    # write out tensorflow checkpoint & inference graph (from MH's "MNIST classification with TensorFlow and Xilinx DNNDK")
#save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fine.ckpt", global_step=0))
if __name__=='__main__':
    keras2tfpb()

