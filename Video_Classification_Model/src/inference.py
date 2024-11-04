import os
import timeit
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import pandas as pd
import matplotlib
from utils.utils import save_test_duration
matplotlib.use('agg')
import sklearn

def inference(model_savepath, x_test):
    start_time = time.time()

    model = keras.models.load_model(model_savepath)
    
    y_pred = model.predict(x_test)
    y_pred_numbered = np.argmax(y_pred, axis=1)

    test_duration = time.time() - start_time
    # save_test_duration('test_duration.csv', test_duration)
    
    return y_pred_numbered

def inference_new_data(model_savepath, data_path):
    start = timeit.default_timer()
    
    possible_categories = ['Beauty & Fashion', 'Education', 'Entertainment ', 
                           'Knowledge', 'Music', 'News', 'Sports', 
                           'Technology', 'Cooking & Health', 'Game', 'Movie']

    x_test = pd.read_csv(os.path.join(data_path, 'test_new.csv'))

    x_test = x_test.iloc[:100, :3000] 
    
    y_pred_numbered = inference(model_savepath, x_test)
    print(y_pred_numbered)

    print(list(map(lambda x: possible_categories[x], y_pred_numbered)))

    end = timeit.default_timer()
    print('Running time: %s Seconds' % (end - start))
    
