import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def selectFeature(X_old, X_new, y_old, draw_figure=False, batch_num=0):

        # compute feature importance score
        params = {'num_leaves': 64, 
                  'feature_fraction':0.8, 
                  'bagging_fraction':0.8, 
                  'bagging_freq': 1,
                  'boosting_type':'rf', 
                  'objective':'binary', 
                  'min_data_in_leaf': 10, 
                  'num_threads':20, 
                  'verbose': -1
                  }
        '''
        params = {'n_estimators': 100, 
                  'learning_rate': 0.01, 
                  'num_leaves': 40, 
                  'feature_fraction':0.8, 
                  'bagging_fraction':0.8, 
                  'bagging_freq': 1,
                  'boosting_type':'gbdt', 
                  'objective':'binary', 
                  'min_data_in_leaf': 10, 
                  'num_threads':20, 
                  'verbose': -1
                  }
        '''
        model = lgb.train(params, lgb.Dataset(X_old, y_old), num_boost_round=50)
        feat_score = model.feature_importance(importance_type='gain')
        
        # compute drifting score
        # downsample
        if (X_old.shape[0] > X_new.shape[0]):
            X_old, _ = train_test_split(X_old, train_size=X_new.shape[0], random_state=0)
        elif (X_old.shape[0] < X_new.shape[0]):
            X_new, _ = train_test_split(X_new, train_size=X_old.shape[0], random_state=0)
        # create training data
        X = pd.concat([X_old, X_new], axis=0, ignore_index=True)
        y = np.concatenate([np.zeros(X_old.shape[0]), np.ones(X_new.shape[0])])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        # train classifier
        drift_score = []
        for col in X.columns:
            try:
                clf = DecisionTreeClassifier(max_leaf_nodes=32).fit(X_train[[col]].fillna(0.).astype(float), y_train)
                drift_score.append(roc_auc_score(y_test, clf.predict_proba(X_test[[col]].fillna(0.).astype(float))[:, 1]))
            except:
                drift_score.append(0.)
        drift_score = np.array(drift_score)
        print (drift_score)

        remove_condition = (drift_score >= 0.75) & (feat_score < np.median(feat_score))
        remain_condition = ~remove_condition
        print ('remove the following features')
        print (X.columns[remove_condition])
        use_index = np.arange(len(drift_score))[remain_condition]

        plt.figure()
        plt.scatter(drift_score[remain_condition], feat_score[remain_condition], label='use')
        plt.scatter(drift_score[remove_condition], feat_score[remove_condition], label='remove')
        plt.legend()
        plt.savefig('dist_%d.png' %(batch_num))
        plt.close()

        return use_index, drift_score