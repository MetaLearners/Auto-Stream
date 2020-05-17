'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import os
#os.system('pip3 install lightgbm==2.2.2')
#os.system('pip3 install hyperopt')

import pandas as pd
import pickle
import numpy as np
import scipy
from os.path import isfile
import os
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from mlbox.preprocessing import *
from mlbox.optimisation import *
#from mlbox.prediction import *

import matplotlib.pyplot as plt

from predictor import *
from preprocess import *
from utils import *


class Model:
    def __init__(self, data_info, time_info):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        
        self.batch_num = 0

        self.data_memory = []

        self.data_preprocess = DataPreprocess(drop_ratio=0.9)
        self.feature_engineering = FeatureEngineering()

        self.first_sample = True
        self.imbalance_threshold = 0.1
        self.imbalance_ratio = 3.
        self.min_sample_per_batch = 100000

        self.score = []
        self.mode = 'static'


    def fit(self, F, y, data_info, time_info):

        y = y.ravel()

        if (self.batch_num > 0):
            print ('=== evaluate batch %d ===' %(self.batch_num))
            auc = 2 * roc_auc_score(y, self.y_pred) - 1
            print ('clf prediction auc: %.4f' %(auc))
            self.score.append(auc)
            print (self.score)
            print ('average score all the time: ', np.array(self.score).mean())

        # preprocess and feature engineering
        if (self.batch_num == 0):
            F = self.data_preprocess.transform(F, data_info['loaded_feat_types'][0], data_info['loaded_feat_types'][3], data_info['loaded_feat_types'], is_train=True)
            try:
                F['CAT'] = F['CAT'].astype(str)
            except:
                F['CAT'] = pd.DataFrame()
            #F = self.feature_engineering.transform(F, y, is_train=True)
            #self.cat_info = F['cat_info'].copy()
            self.F = F

        feat_block = ['numerical', 'CAT', 'time']

        # perform 1st subsample
        if self.first_sample:
            if y.shape[0] > self.min_sample_per_batch:
                class_ratio = y.mean()
                class_ratio = min(class_ratio, 1. - class_ratio)
                if (class_ratio < self.imbalance_threshold):
                    index = majority_downsample(y, self.imbalance_ratio)
                else:
                    index = stratified_downsample(y, self.min_sample_per_batch / y.shape[0])
                y = y[index]
                for feat in feat_block:
                    if feat in self.F:
                        if (self.F[feat].shape[0] != 0):
                            self.F[feat] = self.F[feat].iloc[index, :].reset_index(drop=True)
                print('After 1st subsample', y.shape, y.mean())

        # store current batch of data
        self.data_memory.append([self.F, y])


    def autoSklearnModel(self):

        feat_block = ['numerical', 'time', 'CAT']

        if (self.batch_num == 1):
            self.X_train = pd.concat([self.data_memory[0][0][feat] for feat in feat_block if feat in self.F], axis=1)
            self.y_train = pd.Series(self.data_memory[0][1], index=self.X_train.index).astype(int)
            self.X_test = pd.concat([self.F[feat] for feat in feat_block if feat in self.F], axis=1)

            self.data = {'train': self.X_train, 'target': self.y_train, 'test': self.X_test}
            space = {
                    'ne__categorical_strategy' : {"search":"choice", "space":['<NULL>']}, 
                    'ce__strategy' : {'space' : ['label_encoding']},

                    'fs__strategy' : {'space' : ['variance', 'rf_feature_importance']},
                    'fs__threshold': {'search' : 'choice', 'space' : [0.1, 0.2, 0.3]},

                    'est__strategy' : {"space" : ["LightGBM"]},
                    'est__num_leaves' : {"search" : "choice", "space" : [20, 30, 40, 50, 60]},
                    'est__n_estimators' : {"search" : "choice", "space" : [500, 1000, 1500, 2000]}, 
                    'est__learning_rate' : {"search" : "choice", "space" : [0.01]}, 
                    'est__min_data_in_leaf' : {"search" : "choice", "space" : [2, 5, 10, 20]}, 
                    'est__objective' : {"space" : ['binary']},
                    'est__metric' : {"space" : ['auc']},
                    'est__n_jobs' : {'search': 'choice', 'space': [20]}
                    }
            self.opt = Optimiser(scoring='roc_auc')
            self.best = self.opt.optimise(space, self.data, max_evals=50)
            self.clf = Predictor()
            y_pred = self.clf.fit_predict(self.best, self.data)
        else:
            self.X_test = pd.concat([self.F[feat] for feat in feat_block if feat in self.F], axis=1)
            self.data['test'] = self.X_test
            if (self.mode == 'static'):
                y_pred = self.clf.predict(self.data)
            elif (self.mode =='recent'):
                self.X_train = pd.concat([self.data_memory[-1][0][feat] for feat in feat_block if feat in self.data_memory[-1][0]], axis=1)
                self.y_train = pd.Series(self.data_memory[-1][1], index=self.X_train.index).astype(int)
                self.data['train'] = self.X_train
                self.data['target'] = self.y_train
                y_pred = self.clf.fit_predict(self.best, self.data)
            elif (self.mode == 'all'):
                self.X_train_new = pd.concat([self.data_memory[-1][0][feat] for feat in feat_block if feat in self.data_memory[-1][0]], axis=1)
                self.y_train_new = pd.Series(self.data_memory[-1][1], index=self.X_train_new.index).astype(int)
                self.X_train = pd.concat([self.X_train, self.X_train_new], axis=0, ignore_index=True)
                self.y_train = pd.concat([self.y_train, self.y_train_new], axis=0, ignore_index=True)
                self.data['train'] = self.X_train
                self.data['target'] = self.y_train
                y_pred = self.clf.fit_predict(self.best, self.data)

        return y_pred


    def predict(self, F, data_info, time_info):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return random values...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. 
        The function predict eventually can return probabilities or continuous values.
        '''
        self.batch_num += 1

        F = self.data_preprocess.transform(F, data_info['loaded_feat_types'][0], data_info['loaded_feat_types'][3], data_info['loaded_feat_types'])
        #F = self.feature_engineering.transform(F, None, is_train=False)
        try:
            F['CAT'] = F['CAT'].astype(str)
        except:
            F['CAT'] = pd.DataFrame()
        self.F = F

        self.y_pred = self.autoSklearnModel()

        return self.y_pred


    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))


    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self