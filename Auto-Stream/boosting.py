import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
import pandas as pd
import time

from utils import *


def select_feature_by_importance(feature_new, feature_old, label, 
                                 importance_type='split', scale=1., 
                                 use_all=False, valid_fold=3, 
                                 max_select_num=50):
    
    global auc_old, auc_new

    if (label.mean() < 0.5):
        minor_class = 1
    else:
        minor_class = 0
    
    if ((label == minor_class).sum() <= 1000):
        train_size, test_size = 0.5, 0.5
        reuse_minor = True
    else:
        train_size, test_size = 0.75, 0.25
        reuse_minor = False

    feature = pd.concat([feature_old, feature_new], axis=1, copy=False)
    
    skf = StratifiedKFold(n_splits=int(1 / test_size), random_state=0, shuffle=True)
    for index_1, index_2 in skf.split(np.zeros(label.shape[0]), label):
        train_idx, valid_idx = index_1, index_2
        break

    sub_idx = majority_downsample(label[train_idx], frac=3.0)
    train_idx = train_idx[sub_idx]

    valid_idx_list = []
    if reuse_minor:
        for i in range(valid_fold):
            sub_idx = majority_downsample(label[valid_idx], frac=3.0, seed=i)
            valid_idx_list.append(valid_idx[sub_idx])
    else:
        skf = StratifiedKFold(n_splits=valid_fold, random_state=0, shuffle=True)
        for _, sub_idx in skf.split(np.zeros(valid_idx.shape[0]), label[valid_idx]):
            valid_idx_list.append(valid_idx[sub_idx])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt', 
        'objective': 'binary', 
        'metric': 'auc', 
        'num_leaves': 31, 
        'learning_rate': 0.01, 
        'feature_fraction': 1.0, 
        #'bagging_fraction': self.hyper_param["subsample"], 
        #'bagging_freq': 1, 
        'min_data_in_leaf': 10, 
        'lambda_l1': 0.1, 
        'lambda_l2': 0.1, 
        #'top_rate': 0.1, 
        #'other_rate': 0.05, 
        'num_threads': 20, 
        'verbose': -1
    }

    def evaluate(X, y):

        X_train, y_train = X.iloc[train_idx, :].reset_index(drop=True), y[train_idx]
        train_data = lgb.Dataset(X_train, y_train)
        
        valid_data = []
        for i in range(valid_fold):
            valid_data.append(lgb.Dataset(X.iloc[valid_idx_list[i], :].reset_index(drop=True), y[valid_idx_list[i]]))
        
        clf = lgb.train(params, train_data, num_boost_round=500, 
                        valid_sets=valid_data, 
                        early_stopping_rounds=100, verbose_eval=100)
        
        score = 0.
        for i in range(valid_fold):
            score += roc_auc_score(y[valid_idx_list[i]], clf.predict(X.iloc[valid_idx_list[i], :].reset_index(drop=True)))
        score /= valid_fold

        return clf, score

    print ('evaluate with all new features')
    clf, auc_score = evaluate(feature, label)

    # only need to calculate the validation auc on baseline features
    if (feature_new.shape[1] == 0):
        auc_old = auc_score
        return auc_old

    # reserve all cat_origin features
    if use_all:
        auc_new = auc_score
        print ('validation auc: %.4f, improvement: %.4f' %(auc_new, auc_new - auc_old))
        if (auc_new - auc_old > 0):
            auc_old = auc_new
            return auc_new, feature_new.columns
        else:
            return auc_new, []

    # select features based on feature importance
    feat_score = clf.feature_importance(importance_type=importance_type)
    score_threshold = feat_score.mean() * scale
    feat_new_score = feat_score[feature_old.shape[1]:]
    selected_features = feature_new.columns[feat_new_score > score_threshold]

    # truncate the selected features
    if (len(selected_features) > max_select_num):
        order = np.argsort(-feat_new_score)[:max_select_num]
        order = np.sort(order)
        selected_features = feature_new.columns[order]

    # no new feature is selected
    if (len(selected_features) == 0):
        print ('no feature is selected based on importance score')
        return auc_old, selected_features

    '''
    # do not need to retrain if the auc has improved with all new features
    if (auc_score > auc_old):
        auc_enw = auc_score
        print ('validation auc: %.4f, improvement: %.4f' %(auc_new, auc_new - auc_old))
        auc_old = auc_new
        return auc_new, selected_features
    '''

    # retrain with old features and selected new features
    # decide whether or not to use this type of features based on auc improvement
    print ('evaluate with selected new features')
    feature = pd.concat([feature_old, feature_new[selected_features]], axis=1, copy=False)
    clf, auc_score = evaluate(feature, label)

    # TODO: epsilon?
    auc_new = auc_score
    print ('validation auc: %.4f, improvement: %.4f' %(auc_new, auc_new - auc_old))
    if (auc_new - auc_old > 0):
        auc_old = auc_new
        return auc_new, selected_features
    else:
        return auc_new, []


class GBM:

    def __init__(self, hyper_tune=False):

        self.hyper_tune = hyper_tune
        self.hyper_param = []
        
        self.max_sample = 30000
        self.test_ratio = 0.25

        self.num_leaves = [30, 40, 50, 60]
        #self.num_leaves = [60]
        self.min_data_in_leaf = [2, 5, 10, 20]
        #self.min_data_in_leaf = [1]

        self.max_train_sample = 500000


    def setParams(self, config):

        params = {'num_leaves': 40, 
                  'min_data_in_leaf': 10, 
                  'learning_rate': 0.01, 
                  'feature_fraction':0.8, 
                  'bagging_fraction':0.8, 
                  'bagging_freq':2, 
                  'boosting_type':'gbdt', 
                  'objective':'binary', 
                  #'metric':'auc', 
                  'metric':'binary_logloss', 
                  'num_threads':20, 
                  'lambda_l1': 0.01, 
                  'lambda_l2': 0.01, 
                  'seed': 0, 
                  'verbose': -1
                  }
        params.update(config)
        return params


    def evaluateParams(self, config):

        params = self.setParams(config)
        clf = lgb.train(params, self.train_set, num_boost_round=2000, 
            valid_sets=[self.valid_set], early_stopping_rounds=50, verbose_eval=False)
        score = roc_auc_score(self.y_valid, clf.predict(self.X_valid))
        return score


    def hyperTune(self, X, y):
        
        if len(y) > self.max_sample:
            index = stratified_downsample(y, float(self.max_sample) / len(y))
            XTune, yTune = X.iloc[index, :].reset_index(drop=True), y[index]
        else:
            XTune, yTune = X, y
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(XTune, yTune, test_size=self.test_ratio, random_state=42, stratify=yTune)
        self.train_set = lgb.Dataset(self.X_train, self.y_train, free_raw_data=False).construct()
        self.valid_set = lgb.Dataset(self.X_valid, self.y_valid, free_raw_data=False).construct()

        seed = 0
        score, config = np.zeros(16), []
        for i, num_leaves in enumerate(self.num_leaves):
            for min_data_in_leaf in self.min_data_in_leaf:
                print (seed)
                params = {'num_leaves': num_leaves, 'min_data_in_leaf': min_data_in_leaf, 'seed': seed}
                score[seed] = self.evaluateParams(params)
                config.append(params)
                seed += 1
        order = np.argsort(-score)
        print (score[order])
        self.hyper_param = [config[order[i]] for i in range(4)]
        print (self.hyper_param)


    def fit(self, X, y):
        
        if (len(self.hyper_param) == 0):
            if self.hyper_tune:
                self.hyperTune(X, y)
            else:
                self.hyper_param = [{'num_leaves': 40, 'min_data_in_leaf': 10, 'seed': 0}]

        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
        #train_set = lgb.Dataset(X_train, y_train, free_raw_data=False).construct()
        #valid_set = lgb.Dataset(X_valid, y_valid, free_raw_data=False).construct()

        self.models = []
        for i, hp in enumerate(self.hyper_param):

            if (y.shape[0] > self.max_train_sample):
                index = stratified_downsample(y, self.max_train_sample / y.shape[0], seed=i)
                X_sample, y_sample = X.iloc[index, :], y[index]
                X_train, X_valid, y_train, y_valid = train_test_split(X_sample, y_sample, test_size=0.1, random_state=i, stratify=y_sample)
            else:
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)                
            train_set = lgb.Dataset(X_train, y_train, free_raw_data=False).construct()
            valid_set = lgb.Dataset(X_valid, y_valid, free_raw_data=False).construct()

            clf = lgb.train(self.setParams(hp), train_set, num_boost_round=2000, 
                valid_sets=[valid_set], early_stopping_rounds=50, verbose_eval=False)
            self.models.append(clf)
            print (hp, clf.num_trees())


    def cv(self, X, y):

        skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
        params = self.setParams({'num_leaves':40, 'n_estimators':1000, 'learning_rate':0.01})
        cv_pred = np.zeros_like(y)
        for train_index, valid_index in skf.split(X, y):
            clf = lgb.LGBMClassifier(random_state=20)
            clf.set_params(**params)
            clf.fit(X.iloc[train_index, :], y[train_index])
            cv_pred[valid_index] = clf.predict_proba(X.iloc[valid_index, :])[:, 1]
        return cv_pred


    def predict(self, X):

        y_preds = np.zeros([X.shape[0], len(self.models)])
        for i, clf in enumerate(self.models):
            y_preds[:, i] = clf.predict(X)
        return y_preds.mean(axis=1)