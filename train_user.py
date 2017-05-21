# -*- coding:utf-8 -*-

import numpy as np
from gen_feat_user import make_train_set
from gen_feat_user import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb

import logging
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')
path='/Users/mac/Desktop/jd-master/jdata/'


def xgboost_make_submission():

    train_start_date = '2016-02-01'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-06'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    #ratio = float(np.sum(label==0))/np.sum(label==1)
    logging.info(u'存储训练数据集...')
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    logging.info(u'分割数据集...')


    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 'lambda':100,
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 400
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)

    #logging.info(u'保存模型...')
    #bst.save_model(path+'model/xgb.model')
    logging.info(u'保存特征分数和特征信息...')
    feature_score = bst.get_fscore()

    for key in feature_score:
        feature_score[key] = [feature_score[key]]
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    with open(path+'model/feature_score.csv','w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date,)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data)
    y = bst.predict(sub_trainning_data)

    sub_user_index = pd.DataFrame(sub_user_index)
    sub_user_index['label'] = y
    sub_user_index.to_csv(path+'sub/pred.csv')



def xgboost_cv():
    train_start_date = '2016-03-11'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 400
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain,evallist)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    test = xgb.DMatrix(sub_trainning_date)
    y = bst.predict(test)

    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y
    y_true['label'] = label
    report(pred, y_true)


if __name__ == '__main__':
    #xgboost_cv()
    logging.info(u'程序开始...')
    xgboost_make_submission()
