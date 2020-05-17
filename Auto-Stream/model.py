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
from sklearn.linear_model import SGDClassifier
from category_encoders.ordinal import OrdinalEncoder

import matplotlib.pyplot as plt

from preprocess import *
from boosting import *
from utils import *
from adaptation import *
from deepfm import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class Model:
    def __init__(self, data_info, time_info):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.batch_num = 0
        self.max_window_size = 10
        self.window_size = 0
        self.data_memory = []
        self.window_score = np.zeros(self.max_window_size + 1, dtype=float)

        self.data_preprocess = DataPreprocess(drop_ratio=0.9)
        self.feature_engineering = FeatureEngineering()

        self.first_sample = False
        self.imbalance_threshold = 0.1
        self.imbalance_ratio = 3.
        
        self.max_train = 1000000

        self.score = []

        self.model_score = {model: 0. for model in ['RF', 'GBDT', 'FM', 'LR']}
        self.model_ensemble = ['FM', 'GBDT']
        #self.model_ensemble = ['GBDT']


    def fit(self, F, y, data_info, time_info):

        y = y.ravel()

        # evaluate
        if (self.batch_num > 0):
            print ('=== evaluate batch %d ===' %(self.batch_num))
            print ('train on the last %d batches with %s' %(self.window_size, '+'.join(self.model_ensemble)))
            auc = 2 * roc_auc_score(y, self.y_pred) - 1
            print ('prediction auc: %.4f' %(auc))
            self.score.append(auc)
            print (self.score)
            print ('average score all the time: ', np.array(self.score).mean())

        # adapt window size

        if (self.window_size > 1):
            for batch in self.monitor_pred:
                score = roc_auc_score(y, self.monitor_pred[batch])
                if (self.window_score[batch] == 0):
                    self.window_score[batch] = score
                else:
                    self.window_score[batch] = 0.5 * self.window_score[batch] + 0.5 * score
            self.window_size = np.argmax(self.window_score)
        self.window_size += 1
        self.window_size = min(self.window_size, self.max_window_size)

        # adapt the ensemble
        if (self.batch_num > 0):
            # evaluate each model
            for model in self.model_pred:
                score = roc_auc_score(y, self.model_pred[model])
                print (model, 2 * score - 1)
                if (self.batch_num == 1):
                    self.model_score[model] = score
                else:
                    self.model_score[model] = 0.95 * score + 0.05 * self.model_score[model]
            # select the models for the ensemble
            self.model_ensemble = []
            best_score = max(self.model_score.values())
            for model in self.model_score:
                if (self.model_score[model] >= best_score * 0.98):
                    self.model_ensemble.append(model)
                    print ('use %s for the next batch' %(model))

        # preprocess and feature engineering
        if (self.batch_num == 0):
            F = self.data_preprocess.transform(F, data_info['loaded_feat_types'][0], data_info['loaded_feat_types'][3], data_info['loaded_feat_types'], is_train=True)
            F = self.feature_engineering.transform(F, y, is_train=True)
            self.cat_info = F['cat_info'].copy()
            self.F = F

        # perform 1st subsample
        if self.first_sample:
            class_ratio = y.mean()
            class_ratio = min(class_ratio, 1. - class_ratio)
            if (class_ratio < self.imbalance_threshold):
                index = majority_downsample(y, self.imbalance_ratio)
                y = y[index]
                for feat in self.F:
                    if (len(self.F[feat]) != 0):
                        self.F[feat] = self.F[feat].iloc[index, :].reset_index(drop=True)
            print('After 1st subsample', y.shape, y.mean())

        # store current batch of data
        if (len(self.data_memory) == self.max_window_size):
            del self.data_memory[0]
        self.data_memory.append([self.F, y])

        try:
            for batch in self.data_memory:
                if batch[0]['cat_freq'].shape[0] > 0:
                    batch[0]['cat_freq'] = pd.DataFrame()
        except:
            pass


    def GBDTModel(self):

        print ('=== train GBDT ==')
        if (self.batch_num == 1):
            self.clf = GBM(hyper_tune=True)
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)
        return y_pred


    def deepModel(self):

        print ('=== train NN / DeepFM ==')
        train_num = self.X_train.shape[0]
        X_all = pd.concat([self.X_train, self.X_test], axis=0, ignore_index=True)

        dense_features = list(X_all.columns)
        try:
            sparse_features = list(self.F['CAT'].columns)
        except:
            sparse_features = []

        X_all[dense_features] = MinMaxScaler(feature_range=(0., 1.)).fit_transform(X_all[dense_features])

        if (len(sparse_features) != 0):
            train_cat = pd.concat([batch[0]['CAT'] for batch in self.data_memory[-self.window_size:]], axis=0, ignore_index=True)
            test_cat = self.F['CAT']
            all_cat = pd.concat([train_cat, test_cat], axis=0, ignore_index=True)

            useless_cat = []
            sparse_cat_num = []
            for feat in sparse_features:
                cat_freq = pd.DataFrame(train_cat[feat].value_counts(normalize=False, sort=True, ascending=False, dropna=False))
                unique_num = (cat_freq[feat] >= 100).sum()
                if (unique_num == 0):
                    useless_cat.append(feat)
                else:
                    index = np.ones(cat_freq.shape[0], dtype=int) * unique_num
                    index[:unique_num] = np.arange(unique_num)
                    cat_freq['index'] = index
                    X_all[feat] = all_cat[feat].map(cat_freq['index']).fillna(unique_num).astype(int)
                    sparse_cat_num.append(X_all[feat].nunique())
            for col in useless_cat:
                sparse_features.remove(col)

        print (dense_features)
        print (sparse_features)

        X_train, X_test = X_all.iloc[:train_num, :], X_all.iloc[train_num:, :]

        # downsample training data
        if (train_num > self.max_train):
            X_train, _, y_train, _ = train_test_split(X_train, self.y_train, 
                train_size=self.max_train, random_state=1, stratify=self.y_train)
        else:
            y_train = self.y_train

        X_train_sparse, X_train_dense = X_train[sparse_features], X_train[dense_features]
        X_test_sparse, X_test_dense = X_test[sparse_features], X_test[dense_features]

        if (len(sparse_features) != 0):
            model = DeepFM(16, sparse_cat_num, len(dense_features), [128, 128])
        else:
            model = DeepFM(16, [], len(dense_features), [128, 128])            
        model.fit(X_train_sparse.values, X_train_dense.values, y_train, epoch=100, batch_size=1024, early_stop_epoch=5)
        y_pred = model.predict(X_test_sparse.values, X_test_dense.values, batch_size=1024)

        return y_pred.ravel()


    def RFModel(self):

        print ('=== train random forest ===')
        params = {'num_leaves': 64, 
                  'feature_fraction':0.8, 
                  'bagging_fraction': min(self.max_train / len(self.y_train), 0.8), 
                  'bagging_freq': 1, 
                  'boosting_type':'rf', 
                  'objective':'binary', 
                  'min_data_in_leaf': 10, 
                  'num_threads':20, 
                  'verbose': -1
                  }
        model = lgb.train(params, lgb.Dataset(self.X_train, self.y_train), num_boost_round=50)
        return model.predict(self.X_test)


    def linearModel(self):
        
        print ('=== train linear model ===')
        model = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0., 
            max_iter=1000, tol=0.001, shuffle=True, verbose=0, random_state=2020, 
            learning_rate='adaptive', eta0=0.01, 
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, 
            class_weight=None)
        self.X_train = MinMaxScaler().fit_transform(self.X_train)
        model.fit(self.X_train, self.y_train)
        return model.predict_proba(self.X_test)[:, 1]


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
        F = self.feature_engineering.transform(F, None, is_train=False)
        self.F = F

        feat_block = ['numerical', 'cat_freq', 'time', 'num_mean_groupby_cat', 
                      'cat_nunique_groupby_cat', 'time_delta_groupby_cat', 'cat_origin', 
                      'cat_combine']
        
        # co-encoding of categorical features
        try:
            for col in self.F['CAT'].columns:
                self.cat_info[col] = self.cat_info[col].add(self.F['cat_info'][col], fill_value=0)
                cat_info = self.cat_info[col] / self.cat_info[col].sum()
                for batch in self.data_memory[-self.window_size:]:
                    batch[0]['cat_freq'][col + '_freq'] = batch[0]['CAT'][col].map(cat_info)
                self.F['cat_freq'][col + '_freq'] = self.F['CAT'][col].map(cat_info)
        except:
            pass

        X_train = []
        for batch in self.data_memory:
            X_train.append(pd.concat([batch[0][feat] for feat in feat_block if feat in self.F], axis=1))
        self.X_train = pd.concat(X_train[-self.window_size:], axis=0, ignore_index=True)
        self.y_train = np.concatenate([batch[1] for batch in self.data_memory[-self.window_size:]])

        self.X_test = pd.concat([self.F[feat] for feat in feat_block if feat in self.F], axis=1)

        if (self.X_train.shape[1] > 10):
            feature_index, _ = selectFeature(self.X_train, self.X_test, self.y_train, draw_figure=False, batch_num=self.batch_num)
            self.X_train = self.X_train.iloc[:, feature_index]
            self.X_test = self.X_test.iloc[:, feature_index]

        self.model_pred = {}
        self.model_pred['GBDT'] = self.GBDTModel()
        self.model_pred['RF'] = self.RFModel()
        self.model_pred['FM'] = self.deepModel()
        self.model_pred['LR'] = self.linearModel()
        self.y_pred = sum([self.model_pred[m] for m in self.model_ensemble]) / len(self.model_ensemble)

        if (self.window_size > 1):
            self.monitor_pred = {}
            params = {'n_estimators': 50, 
                      'num_leaves': 64, 
                      'feature_fraction':0.8, 
                      'bagging_fraction': min(self.max_train / len(self.y_train), 0.8), 
                      'bagging_freq': 1,
                      'boosting_type':'rf', 
                      'objective':'binary', 
                      'min_data_in_leaf': 10, 
                      'num_threads':10, 
                      'verbose': -1
                      }
            for i in range(1, len(self.data_memory) + 1):
            #for i in range(1, self.window_size + 1):
                X_train_part = pd.concat(X_train[-i:], axis=0, ignore_index=True)
                y_train_part = np.concatenate([batch[1] for batch in self.data_memory[-i:]])
                model = lgb.train(params, lgb.Dataset(X_train_part, y_train_part))
                self.monitor_pred[i] = model.predict(self.X_test)

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