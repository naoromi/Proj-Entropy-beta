import numpy as np
import pandas as pd
import sklearn
import random
import tensorflow as tf
from tensorflow import keras
import argparse
from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# from utils.utils import generate_results_csv
# from utils.utils import create_directory
# from utils.utils import read_dataset
# from utils.utils import transform_mts_to_ucr_format
# from utils.utils import visualize_filter
# from utils.utils import viz_for_survey_paper
# from utils.utils import viz_cam
# from utils.utils import save_test_duration
# from utils.utils import calculate_metrics
# from utils.utils import read_all_datasets
import os
import numpy as np
import sys
import sklearn
import os
import timeit
import tensorflow.keras as keras
import numpy as np
import time
import pandas as pd
import matplotlib
matplotlib.use('agg')
import sklearn




def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from .models import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from .models import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from .models import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from .models import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from .models import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from .models import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from .models import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from .models import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn_tsc':  # Time-CNN
        from .models import cnn_tsc
        return cnn_tsc.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from .models import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


#use time_size as input
def fit_classifier(data_path, output_path, classifier_name):
    #x_train, x_test, y_train, y_test = train_test_split(train, train_labels)

    x_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    # x_test = pd.read_csv(os.path.join(data_path,'test_new.csv'))
    
    y_train = pd.read_csv(os.path.join(data_path, "Y_train.csv"))
    # y_test = pd.read_csv(os.path.join(data_path,'test_labels.csv'))

    # split train into train and validation
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    # x_train = x_train.iloc[:, :300] 
    # y_train = y_train[:] 
    # x_test = x_test.iloc[:, :300] 
    # y_test = y_test[:] 


    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary (label转化为2进制)
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    output_directory = output_path
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def train_model(input_path, model_save_path):
    setup_seed(100)
    os.makedirs(model_save_path)
    
    fit_classifier(input_path,
                   model_save_path,
                   classifier_name = "resnet")
    
    '''
    # become command line tool with arguments
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='', help='Path to the dataset.')
    parser.add_argument('--output_directory', type=str, default='', help='Output directory for saving the models.')
    parser.add_argument('--classifier_name', type=str, default='resnet', help='Classifier to use. One of "fcn", "mlp", "resnet", "mcnn", "tlenet", "twiesn", "encoder", "mcdcnn", "cnn_tsc", "inception".')

    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    print(args.output_directory)
    
    fit_classifier(data_path = args.data_path , output_path= args.output_directory , classifier_name = args.classifier_name)
    '''


