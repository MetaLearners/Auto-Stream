import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy
from sklearn.metrics import roc_auc_score
import time

from boosting import select_feature_by_importance


# data preprocessing
class DataPreprocess:

    def __init__(self, drop_ratio=0.9):
        '''
        1. split time and numerical features
        2. transform data block into dataframe and correct column names
        3. remove feature columns with too many missing values
        4. impute missing values
        '''
        self.drop_ratio = drop_ratio
        self.use_columns = {}
        self.selected_cat_cols = []

    def transform(self, F, time_feature_dimension, mv_feature_dimension, feat_num, is_train=False):
        '''
        time_feature_dimension: the dimension of time feature
        mv_feature_dimension: the dimension of mv feature
        feat_columns: the columns index for each type of feature in the source data
        '''

        # split time and numerical features
        if (time_feature_dimension != 0):
            F['time'] = F['numerical'][:, :time_feature_dimension]
            F['numerical'] = F['numerical'][:, time_feature_dimension:]

        # transform data block into dataframe and correct column names
        if (time_feature_dimension != 0):
            F['time'] = pd.DataFrame(F['time'], columns=['time_' + str(col) for col in range(feat_num[0])])
        F['numerical'] = pd.DataFrame(F['numerical'], columns=['num_' + str(col) for col in range(feat_num[1])])
        try:
            F['CAT'] = F['CAT'].add_prefix('cat_')
        except:
            pass
        if (mv_feature_dimension != 0):
            F['MV'] = F['MV'].add_prefix('mv_')

        # remove feature columns with too many missing values
        feat_types = ['numerical', 'CAT']
        if (len(F['CAT']) == 0):
            feat_types.remove('CAT')
        if (time_feature_dimension != 0):
            feat_types.append('time')
        if (mv_feature_dimension != 0):
            feat_types.append('MV')

        for feat_type in feat_types:
            # determine the dropping columns on the training batch
            if feat_type not in self.use_columns:
                missing_ratio = F[feat_type].isnull().mean(axis=0).values
                self.use_columns[feat_type] = np.arange(F[feat_type].shape[1])[missing_ratio < self.drop_ratio]
            
            F[feat_type] = F[feat_type].iloc[:, self.use_columns[feat_type]]

        # combine cat and mv features
        #if (mv_feature_dimension != 0):
        #    F['CAT'] = pd.concat([F['CAT'], F['MV']], axis=1)

        # impute missing values
        try:
            F['CAT'] = F['CAT'].fillna('-1')
        except:
            pass
        F['numerical'] = F['numerical'].fillna(0.)

        #cols = F['numerical'].columns
        #F['numerical'][cols] = MinMaxScaler().fit_transform(F['numerical'][cols].values)

        if (time_feature_dimension != 0):
            F['time'] = F['time'].fillna(F['time'].mean(axis=0))
            F['time'] -= F['time'].min(axis=0)
            #cols = F['time'].columns
            #F['time'][cols] = MinMaxScaler().fit_transform(F['time'][cols].values)

        # select cat columns
        try:
            if is_train:
                sample_num = F['CAT'].shape[0]
                for col in F['CAT'].columns:
                    cat_num = F['CAT'][col].nunique()
                    if ((cat_num > 1) and (cat_num < sample_num / 2)):
                        self.selected_cat_cols.append(col)
                    else:
                        print ('remove %s with %d categories' %(col, cat_num))

            F['CAT'] = F['CAT'][self.selected_cat_cols]
            F['cat_freq'] = pd.DataFrame()

            if is_train:
                global cat_feat_num
                cat_feat_num = 0
                for col in F['CAT'].columns:
                    if (col.split('_')[0] == 'cat'):
                        cat_feat_num += 1

        except:
            pass

        return F


class FeatureEngineering:

    def __init__(self):
        
        self.transformer = {
            'cat_origin': CatOrigin(max_cat_num=32), 
            'num_mean_groupby_cat': CatNumAggregate(min_cat_num=100, max_cat_ratio=2), 
            'cat_nunique_groupby_cat': CatNuniqueGroupbyCat(min_cat_num=100, max_cat_ratio=10), 
            'time_delta_groupby_cat': CatTimeAggregate(min_cat_num=1000, max_cat_ratio=5), 
            'cat_combine': CatCombine(min_cat_num=100, max_cat_num=10000)
        }

    def transform(self, F, y, is_train=False):

        time_spent = []

        # category count
        try:
            start_time = time.time()
            X_cat = F['CAT']
            F['cat_info'] = {}
            for col in X_cat.columns:
                cat_info = X_cat[col].value_counts(sort=False, dropna=False)
                F['cat_info'][col] = cat_info
            end_time = time.time()
            time_spent.append(('cat_count', end_time - start_time))
        except:
            pass

        # high-order feature engineering
        feat_types = ['cat_nunique_groupby_cat', 'time_delta_groupby_cat', 'cat_combine']
        if 'time' not in F:
            feat_types.remove('time_delta_groupby_cat')
        #feat_types = []
        
        # set up the baseline for feature selection
        if is_train:
            # generate cat/mv freq features
            X_cat_freq = pd.DataFrame()
            for col in F['CAT'].columns:
                X_cat_freq[col + '_freq'] = F['CAT'][col].map(F['cat_info'][col] / F['cat_info'][col].sum())
            # concat basic features
            X_baseline = pd.concat([F['numerical'], X_cat_freq], axis=1)
            if 'time' in F:
                X_baseline = pd.concat([X_baseline, F['time']], axis=1)
            # train and validate on the training set
            auc_baseline = select_feature_by_importance(pd.DataFrame(), X_baseline, y)
            print ('validation auc with basic features: ', auc_baseline)
        
        for feat in feat_types:
            print ('processing %s features' %(feat))
            transformer = self.transformer[feat]
            start_time = time.time()
            if is_train:
                F[feat] = transformer.fit_transform(F, y, X_baseline)
                X_baseline = pd.concat([X_baseline, F[feat]], axis=1)
            else:
                F[feat] = transformer.transform(F)
            end_time = time.time()
            time_spent.append((feat, end_time - start_time))    

        for feat in time_spent:
            print ('%s: %.2f seconds' %(feat[0], feat[1]))
        
        return F


class CatNumAggregate:

    def __init__(self, min_cat_num=100, max_cat_ratio=2):
        self.min_cat_num = min_cat_num
        self.max_cat_ratio = max_cat_ratio
        self.cat_used = []
        self.mean_columns = {}
        self.subtract_columns = {}

    def fit_transform(self, F, label, X_old):

        try:
            self.X_cat = F['CAT'].iloc[:, :cat_feat_num]
            self.X_num = F['numerical']

            # select categorical columns used for aggregation
            cat_count = self.X_cat.nunique().sort_values(ascending=False)
            self.cat_used = list(cat_count.index[((cat_count >= self.min_cat_num) & \
                            (cat_count <= self.X_cat.shape[0] // self.max_cat_ratio))])
            print ('cat used: ', self.cat_used)

            # aggregate
            X_mean_all, X_subtract_all = [], []
            for cat in self.cat_used:
                X = pd.concat([self.X_cat[[cat]], self.X_num], axis=1)
                X_mean = X.groupby(cat).mean()
                X_mean = X_mean.rename(columns={col: col + '_mean_groupby_' + cat for col in X_mean.columns})
                X_mean = X[[cat]].merge(X_mean.reset_index(), how='left', on=cat).iloc[:, 1:]
                X_subtract = pd.DataFrame(self.X_num.values - X_mean.values, columns=[col + '_subtract' for col in X_mean.columns])
                X_mean_all.append(X_mean)
                X_subtract_all.append(X_subtract)
            X_mean_all = pd.concat(X_mean_all, axis=1)
            X_subtract_all = pd.concat(X_subtract_all, axis=1)
            X_new = pd.concat([X_mean_all, X_subtract_all], axis=1)

            # feature selection
            auc_score, selected_features = select_feature_by_importance(X_new, X_old, label, importance_type='gain', scale=1.)
            X_new = X_new[selected_features]

            print ('selected features: ', selected_features)
            for col in selected_features:
                num_col, cat_col = col.split('_mean_groupby_')
                if '_subtract' in cat_col:
                    cat_col = cat_col.split('_subtract')[0]
                    try:
                        self.subtract_columns[cat_col].append(num_col)
                    except:
                        self.subtract_columns[cat_col] = [num_col]
                else:
                    try:
                        self.mean_columns[cat_col].append(num_col)
                    except:
                        self.mean_columns[cat_col] = [num_col]

            return X_new

        except:
            return pd.DataFrame()


    def transform(self, F):

        self.X_cat = F['CAT'].iloc[:, :cat_feat_num]
        self.X_num = F['numerical']
        X_new = []

        for cat in self.mean_columns:
            X = pd.concat([self.X_cat[[cat]], self.X_num[self.mean_columns[cat]]], axis=1)
            X_mean = X.groupby(cat).mean()
            X_mean = X_mean.rename(columns={col: col + '_mean_groupby_' + cat for col in X_mean.columns})
            X_mean = X[[cat]].merge(X_mean.reset_index(), how='left', on=cat).iloc[:, 1:]
            X_new.append(X_mean)

        for cat in self.subtract_columns:
            X = pd.concat([self.X_cat[[cat]], self.X_num[self.subtract_columns[cat]]], axis=1)
            X_mean = X.groupby(cat).mean()
            X_mean = X_mean.rename(columns={col: col + '_mean_groupby_' + cat for col in X_mean.columns})
            X_mean = X[[cat]].merge(X_mean.reset_index(), how='left', on=cat).iloc[:, 1:]
            X_subtract = self.X_num[self.subtract_columns[cat]].values - X_mean.values
            X_subtract = pd.DataFrame(X_subtract, columns=[col + '_mean_groupby_' + cat + '_subtract' for col in self.subtract_columns[cat]])
            X_new.append(X_subtract)

        try:
            X_new = pd.concat(X_new, axis=1)
        except:
            X_new = pd.DataFrame()
        return X_new


class CatNuniqueGroupbyCat:

    def __init__(self, min_cat_num=100, max_cat_ratio=10):
        self.min_cat_num = min_cat_num
        self.max_cat_ratio = max_cat_ratio
        self.cat_index = []
        self.use_columns = {}

    def fit_transform(self, F, label, X_old):

        try:
            X_cat = F['CAT'].iloc[:, :cat_feat_num]

            # select feature columns for feature generation
            for col in X_cat.columns:
                cat_num = X_cat[col].nunique()
                if ((cat_num > self.min_cat_num) and (cat_num < (X_cat.shape[0] / self.max_cat_ratio))):
                    self.cat_index.append(col)
            print (self.cat_index)
            if (len(self.cat_index) == 0):
                return pd.DataFrame()

            # generate features
            X_new = []
            for col in self.cat_index:
                cols_grouped = [c for c in self.cat_index if c != col]
                cat_count = X_cat[cols_grouped].groupby(X_cat[col]).nunique()
                cat_count.columns = [c + '_nunique_groupby_' + col for c in cat_count.columns]
                cat_count = cat_count.reset_index()
                X_new.append(X_cat[[col]].merge(cat_count, on=col, how='left').iloc[:, 1:])
            X_new = pd.concat(X_new, axis=1)

            # select generated features by importance
            auc_score, selected_features = select_feature_by_importance(X_new, X_old, label, importance_type='gain', scale=1.)
            X_new = X_new[selected_features]

            print ('selected features: ')
            for col in selected_features:
                print (col)
                cat_1, cat_2 = col.split('_nunique_groupby_')
                try:
                    self.use_columns[cat_2].append(cat_1)
                except:
                    self.use_columns[cat_2] = [cat_1]

            return X_new

        except:
            return pd.DataFrame()


    def transform(self, F):

        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_new = []
        for col in self.use_columns:
            cat_count = X_cat[self.use_columns[col]].groupby(X_cat[col]).nunique()
            cat_count.columns = [c + '_nunique_groupby_' + col for c in cat_count.columns]
            cat_count = cat_count.reset_index()
            X_new.append(X_cat[[col]].merge(cat_count, on=col, how='left').iloc[:, 1:])
        try: 
            X_new = pd.concat(X_new, axis=1)
        except: 
            X_new = pd.DataFrame()
        return X_new


class CatTimeAggregate:

    def __init__(self, min_cat_num=1000, max_cat_ratio=5):
        self.min_cat_num = min_cat_num
        self.max_cat_ratio = max_cat_ratio
        self.cat_index = []
        self.use_columns = {}

    def fit_transform(self, F, label, X_old):

        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_time = F['time']

        # select categorical feature columns for feature generation
        for col in X_cat.columns:
            cat_num = X_cat[col].nunique()
            if ((cat_num > self.min_cat_num) and (cat_num < (X_cat.shape[0] / self.max_cat_ratio))):
                self.cat_index.append(col)
        print (self.cat_index)

        # generate features
        X_new = []
        for time in X_time.columns:

            X = pd.concat([X_time[time], X_cat[self.cat_index]], axis=1)
            X_order_by_time = X.sort_values(time, ascending=True)
            
            for cat in self.cat_index:
                X_group = X_order_by_time[[time, cat]].groupby(cat)
                X_order_by_time[time + '_diff_back_groupby_' + cat] = X_group[time].diff(periods=1)
                X_order_by_time[time + '_diff_forward_groupby_' + cat] = X_group[time].diff(periods=-1)

            X_new.append(X_order_by_time.iloc[:, X.shape[1]:].sort_index())
        X_new = pd.concat(X_new, axis=1)

        # select generated features by importance
        auc_score, selected_features = select_feature_by_importance(X_new, X_old, label, importance_type='gain', scale=1.)
        X_new = X_new[selected_features]

        print ('selected features: ')
        for col in selected_features:
            print (col)
            time, remaining = col.split('_diff_')
            direction, cat = remaining.split('_groupby_')
            try:
                self.use_columns[time].append((direction, cat))
            except:
                self.use_columns[time] = [(direction, cat)]

        return X_new

    def transform(self, F):

        X_time = F['time']
        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_new = pd.DataFrame()
        for time in self.use_columns:
            cat_cols = []
            for feat in self.use_columns[time]:
                if feat[1] not in cat_cols:
                    cat_cols.append(feat[1])
            X = pd.concat([X_time[time], X_cat[cat_cols]], axis=1)
            X_order_by_time = X.sort_values(time, ascending=True)
            for direction, cat in self.use_columns[time]:
                X_group = X_order_by_time[[time, cat]].groupby(cat)
                if (direction == 'back'):
                    X_new[time + '_diff_back_groupby_' + cat] = X_group[time].diff(periods=1).sort_index()
                else:
                    X_new[time + '_diff_forward_groupby_' + cat] = X_group[time].diff(periods=-1).sort_index()
        return X_new


class CatOrigin:

    def __init__(self, max_cat_num=32):
        self.max_cat_num = max_cat_num
        self.col_categories = {}

    def fit_transform(self, F, label, X_old):
        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_new = pd.DataFrame()
        categories_record = {}
        for col in X_cat.columns:
            categories = list(set(X_cat[col].values))
            cat_num = len(categories)
            if ((cat_num <= self.max_cat_num) and (cat_num > 1)):
                try:
                    #F['cat_origin'][col + '_origin'] = X_cat[col].astype(int)
                    X_new[col + '_origin'] = pd.Categorical(X_cat[col])
                    categories_record[col] = categories
                except:
                    pass

        # select generated features by importance
        auc_score, selected_features = select_feature_by_importance(X_new, X_old, label, importance_type='gain', scale=1., use_all=True)
        X_new = X_new[selected_features]

        print ('selected features: ', selected_features)
        for feat in selected_features:
            col = feat.split('_origin')[0]
            self.col_categories[col] = categories_record[col]

        return X_new

    def transform(self, F):
        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_new = pd.DataFrame()
        for col in self.col_categories:
            X_new[col + '_origin'] = pd.Categorical(X_cat[col], categories=self.col_categories[col])
        return X_new


class CatCombine:

    def __init__(self, min_cat_num=100, max_cat_num=10000):
        self.min_cat_num = min_cat_num
        self.max_cat_num = max_cat_num
        self.cat_index = []
        self.selected_combination = []

    def fit_transform(self, F, label, X_old):

        try:
            X_cat = F['CAT'].iloc[:, :cat_feat_num]
            
            # select categorical feature columns for feature generation
            self.cat_index = []
            for col in X_cat.columns:
                cat_num = X_cat[col].nunique()
                if ((cat_num > self.min_cat_num) and (cat_num < self.max_cat_num)):
                    self.cat_index.append(col)
            print (self.cat_index)

            # generate new features
            X_combine = pd.DataFrame()
            X_new = pd.DataFrame()
            for i, cat_1 in enumerate(self.cat_index[:-1]):
                for j, cat_2 in enumerate(self.cat_index[i + 1:]):
                    col = cat_1 + '_' + cat_2 + '_combine'
                    X_combine[col] = list(zip(X_cat[cat_1], X_cat[cat_2]))
                    combine_freq = X_combine[col].value_counts(normalize=True, sort=False).reset_index()
                    combine_freq.columns = [col, col + '_freq']
                    X_new[col + '_freq'] = X_combine[[col]].merge(combine_freq, on=col, how='left')[col + '_freq']

            # select generated features by importance
            auc_score, selected_features = select_feature_by_importance(X_new, X_old, label, importance_type='gain', scale=1.)
            X_new = X_new[selected_features]

            print ('selected features: ')
            for feat in selected_features:
                print (feat)
                name = feat.split('_')
                cat_1 = name[0] + '_' + name[1]
                cat_2 = name[2] + '_' + name[3]
                self.selected_combination.append((cat_1, cat_2))

            return X_new

        except:
            return pd.DataFrame()


    def transform(self, F):
        X_cat = F['CAT'].iloc[:, :cat_feat_num]
        X_new = pd.DataFrame()
        X_combine = pd.DataFrame()
        for cat_1, cat_2 in self.selected_combination:
                col = cat_1 + '_' + cat_2 + '_combine'
                X_combine[col] = list(zip(X_cat[cat_1], X_cat[cat_2]))
                combine_freq = X_combine[col].value_counts(normalize=True, sort=False).reset_index()
                combine_freq.columns = [col, col + '_freq']
                X_new[col + '_freq'] = X_combine[[col]].merge(combine_freq, on=col, how='left')[col + '_freq']
        return X_new