from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets



def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'resnet_pytorch':
        from classifiers import resnet_pytorch
        return resnet_pytorch.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    
    # I suppose we could have other classifiers here.
    

############################################### main

# change this directory for your machine
root_dir = ''
import pandas as pd
def main():
    archive_name = 'UCRArchive_2018'
    dataset_name = 'custom'
    classifier_name = 'resnet_pytorch'
    itr = '_itr_5'

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/'


    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    create_directory(output_directory)
    df_x = pd.read_csv("train-data.csv").values[1:,:]
    x_train=df_x[:3800,:]
    x_val=df_x[3800:,:]
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / x_train.std(axis=1, keepdims=True)
    x_test = (x_val - x_val.mean(axis=1, keepdims=True)) / x_val.std(axis=1, keepdims=True)
    df_y = pd.read_csv("train-labels.csv").values[1:]
    y_train=df_y[:3800]
    y_test=df_y[3800:]
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(df_y.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    x_true = np.argmax(y_train, axis=1)
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, x_true,y_true)

    create_directory(output_directory + '/DONE')
main()