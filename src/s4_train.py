#!/usr/bin/env python
# coding: utf-8

''' This script does:
1. Load features and labels from csv files
2. Train the model
3. Save the model to `model/` folder.
'''

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report
import argparse

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_classifier import ClassifierOfflineTrain



def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config_ej.yaml")
cfg = cfg_all["s4_train.py"]

CLASSES = np.array(cfg_all["classes"])


SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])

DST_MODEL_PATH = par(cfg["output"]["model_path"])

# -- Functions

def train_test_split(X, Y, ratio_of_test_size):
    ''' Split training data by ratio '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Make train/test the same.
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Evaluate accuracy and time cost '''

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.7f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



# -- Main

def run(
        hidden_layer_size=(20,30,40),  # layer structure
        activate = 'relu',  # activate, {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
        solver='adam',  # optimizer, solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
        learning_rate='constant', #{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
        learning_rate_init=0.001,
        max_iter=200, #Maximum number of iterations (determined by ‘tol’).
        tol=1e-4,  # Tolerance for the optimization. 
        early_stopping=False):
    
    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels
    
    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.3) #은지수정: 원본 - 0.3
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Train the model
    print("\nStart training model ...")
    print(hidden_layer_size, activate, solver, learning_rate, learning_rate_init, max_iter, tol, early_stopping)
    model = ClassifierOfflineTrain(hidden_layer_size_=hidden_layer_size, 
                                   activate_=activate, 
                                   solver_=solver, 
                                   learning_rate_=learning_rate, 
                                   learning_rate_init_=learning_rate_init, 
                                   max_iter_=max_iter, 
                                   tol_=tol, 
                                   early_stopping_=early_stopping)
    model.train(tr_X, tr_Y)
    
    # -- Evaluate model
    print("\nStart evaluating model ...")
    evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    print("\nSave model to " + DST_MODEL_PATH)
    with open(DST_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    return model
    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_layer_size', nargs='+', type=int, default=(20,30,40), help='model structure')
    parser.add_argument('--activate', type=str, default='relu', help='{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}')
    parser.add_argument('--solver', type=str, default='adam', help='{‘lbfgs’, ‘sgd’, ‘adam’}')
    parser.add_argument('--learning_rate', '--lr', nargs='+', type=str, default='constant', help='{‘constant’, ‘invscaling’, ‘adaptive’}')
    parser.add_argument('--learning_rate_init', '--lr_init', type=float, default=0.001, help='default:0.001')
    parser.add_argument('--max_iter', type=int, default=200, help='maximum number of iteration')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for the optimization')
    parser.add_argument('--early_stopping', default=False, action='store_true')
    opt = parser.parse_args()
    return opt

def main():
    run(**vars(opt))
    return model


# EJ_loss graph
def loss_curve(model):    
    plt.ylim(0,1)
    plt.plot(model.clf.loss_curve_)
    plt.plot(model.clf.validation_scores_)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()        
        
'''
if __name__ == "__main__":
    main()
'''