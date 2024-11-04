import os
import timeit
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import pandas as pd
import matplotlib
from .utils.utils import save_test_duration
matplotlib.use('agg')
from .utils.utils import calculate_metrics
import sklearn

def evaluate(model_savepath, x_test, y_true, return_df_metrics=True):
    start_time = time.time()
    
    model = keras.models.load_model(model_savepath)
    
    y_pred = model.predict(x_test)
    y_pred_trans = np.argmax(y_pred, axis=1) #预测值转为二进制


    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        print(df_metrics)
        print("################Predict results###########")
        print(y_pred_trans)
        return df_metrics, y_pred
    else:
        test_duration = time.time() - start_time
        save_test_duration('test_duration.csv', test_duration)
        return y_pred

def evaluate_model(model_savepath, data_path):
    start = timeit.default_timer()

    y_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    x_test = pd.read_csv(os.path.join(data_path, 'Y_test.csv'))

    x_test = x_test.iloc[:100, :3000] 
    y_test = y_test[:100] 

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.array((y_test)).reshape(-1, 1))
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()
    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    # how many different values in y_true
    df_metrics, y_pred = evaluate(model_savepath,
                                  x_test, 
                                  y_true,
                                  return_df_metrics=True)
    print(df_metrics)
    
    # print("################True results###########")
    # print(y_true.shape)
    # print(y_true)

    end = timeit.default_timer()
    print('Running time: %s Seconds' % (end - start))
    #show predict result for different types
    class_labels = enc.categories_[0]
    print(class_labels)
    # show df_metrics result for different types
    for i, class_label in enumerate(class_labels):
        print(i, "-", class_label)
        # print("y_pred[y_true == i] Predict results",y_pred[y_true == i],y_pred.shape,y_pred[y_true == i].shape)
        # print("y_true[y_true == i] True results",y_true[y_true == i],y_true.shape,y_true[y_true == i].shape)
        df_metrics = calculate_metrics(y_true[y_true ==i], y_pred[y_true == i], 0.0)
        print(df_metrics)
        print("\n\n")

        # print("################Predict results###########")
        # print(y_pred[y_true == i])
        # print("################True results###########")
        # print(y_true[y_true == i])
