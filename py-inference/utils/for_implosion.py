from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import os
WEIGHT = '../weights'
KERAS_MODEL = 'age_gender_model.hdf5'
def csv_for_implosion(save_csv='implosion.csv'):
    model = load_model(os.path.join(WEIGHT, KERAS_MODEL))
    keras_weights = model.get_weights()
    #num_weights = len(keras_weights)
    num_layers = len(model.layers)
    #print(num_weights, num_layers)
    weights_ = []
    for i in range(num_layers):
     #print(i, keras_weights[i].sum(), keras_weights[i].)
          weights_.append([i, model.layers[i].name, model.weights[i].shape, np.sum(model.weights[i])])


    df = pd.DataFrame(weights_, columns=['layer idx', 'layer name', 'layer shape', 'layer weight sum'])
    df.to_csv(save_csv)

if __name__=='__main__':
    # https://www.jstage.jst.go.jp/article/tjsai/35/3/35_C-JA3/_pdf
    csv_for_implosion(save_csv=os.path.join(WEIGHT, 'implosion.csv'))

