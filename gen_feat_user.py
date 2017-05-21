#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
0.8797
'''
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s: %(levelname)s: %(message)s')
path = '/Users/mac/Desktop/jd-maste/jdata/'


action_1_path = path+"data/JData_Action_201602.csv"
action_2_path = path+"data/JData_Action_201603.csv"
action_3_path = path+"data/JData_Action_201604.csv"
comment_path = path+"data/JData_Comment.csv"
product_path = path+"data/JData_Product.csv"
user_path = path+"data/JData_User.csv"

comment_date = ["2016-03-07", "2016-03-14","2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'18岁以下':
        return 1
    elif age_str == u'19-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

def get_basic_user_feat():
    dump_path = path+'cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path,'rb'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        user['sex'] = user['sex'].fillna(2)

        user['age_sex'] = 10* user['age'].astype(int) + user['sex'].astype(int)
        user['age_lv'] = 10* user['age'].astype(int) + user['user_lv_cd'].astype(int)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        age_sex = pd.get_dummies(user['age_sex'], prefix='age_sex')
        age_lv = pd.get_dummies(user['age_lv'], prefix='age_lv')
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df, age_sex, age_lv], axis=1)
        pickle.dump(user, open(dump_path, 'w'))
    return user


def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action

def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2

def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


def get_actions(start_date, end_date):
    dump_path = path+'cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'r'))
    else:
        action_1 = get_actions_1()
        action_2 = get_actions_2()
        action_3 = get_actions_3()
        actions = pd.concat([action_1, action_2, action_3]) # type: pd.DataFrame
        #actions = pd.concat([action_2, action_3])
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        actions = actions.drop_duplicates()
        #pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_basic_action_feat(start_date, end_date):
    dump_path = path+'cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:

        actions = get_actions(start_date, end_date)


        logging.info(u'时间窗内用户的行为统计特征...')
        type_df = pd.get_dummies(actions['type'], prefix='user_action')
        logging.info(u'时间窗内用户购买种类统计特征...')
        cate_df = pd.get_dummies(actions['cate'], prefix='cate')
        logging.info(u'时间窗内种类和行为的交叉特征...')
        actions['cate_type'] = 10*actions['cate'].astype(int)+actions['type']
        cate_type_df = pd.get_dummies(actions['cate_type'], prefix='cate_type')


        buy_actions = actions[['user_id','type','time']]
        buy_actions = buy_actions[buy_actions['type']==4]
        logging.info(u'时间窗内用户最后购买距离观测日的时长...')
        lastbuy = buy_actions[['user_id', 'time']]
        lastbuy = lastbuy.sort_values('time',ascending=False).groupby(['user_id'],as_index=False).first()
        lastbuy['end_date'] = end_date
        lastbuy['delta'] = pd.to_datetime(lastbuy.end_date) - pd.to_datetime(lastbuy.time)
        lastbuy['delta'] = lastbuy.delta.apply(lambda x: pd.to_timedelta(x).days)

        logging.info(u'时间窗内用户的活跃次数...')
        active_num = actions[['user_id','time']].drop_duplicates().groupby('user_id',as_index=False).count()
        active_num.columns = ['user_id','active_num']

        logging.info(u'时间窗内用户的行为比率...')
        actions = pd.concat([actions['user_id'], type_df, cate_df, cate_type_df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = 1.0*actions['user_action_4'] / actions['user_action_1']
        actions['user_action_2_ratio'] = 1.0*actions['user_action_4'] / actions['user_action_2']
        actions['user_action_3_ratio'] = 1.0*actions['user_action_4'] / actions['user_action_3']
        actions['user_action_5_ratio'] = 1.0*actions['user_action_4'] / actions['user_action_5']
        actions['user_action_6_ratio'] = 1.0*actions['user_action_4'] / actions['user_action_6']

        actions = pd.merge(actions, lastbuy[['user_id','delta']], how='left', on='user_id')
        actions = pd.merge(actions, active_num[['user_id','active_num']], how='left', on='user_id')
        pickle.dump(actions, open(dump_path, 'w'))
    return actions



'''
用户维度的特征
1、行为
2、购买品类
3、品类和行为的交叉特征
4、最后购买距离观测日的天数
5、时间窗内用户的活跃度
'''

def get_accumulate_user_feat(start_date, end_date, day):
    dump_path = path+'cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate']!=11]

        logging.info(u'时间窗 %s 内用户的行为统计特征...' % day)
        type_df = pd.get_dummies(actions['type'], prefix='%sday_user_action' % day)
        logging.info(u'时间窗 %s 内用户购买种类统计特征...' % day)
        cate_df = pd.get_dummies(actions['cate'], prefix='%sday_cate' % day)
        logging.info(u'时间窗 %s 内种类和行为的交叉特征...' % day)
        actions['cate_type'] = 10*actions['cate'].astype(int)+actions['type']
        cate_type_df = pd.get_dummies(actions['cate_type'], prefix='%sday_cate_type' % day)


        buy_actions = actions[['user_id','type','time']]
        buy_actions = buy_actions[buy_actions['type']==4]
        logging.info(u'时间窗 %s 内用户最后购买距离观测日的时长...' % day)
        lastbuy = buy_actions[['user_id', 'time']]
        lastbuy = lastbuy.sort_values('time',ascending=False).groupby(['user_id'],as_index=False).first()
        lastbuy['end_date'] = end_date
        lastbuy['delta'] = pd.to_datetime(lastbuy.end_date) - pd.to_datetime(lastbuy.time)
        lastbuy['%sday_delta' % day] = lastbuy.delta.apply(lambda x: pd.to_timedelta(x).days)

        logging.info(u'时间窗 %s 内用户的活跃次数...' % day)
        active_num = actions[['user_id','time']].drop_duplicates().groupby('user_id',as_index=False).count()
        active_num.columns = ['user_id','%sday_active_num' % day]

        logging.info(u'时间窗 %s 内用户的行为比率...' % day)
        actions = pd.concat([actions['user_id'], type_df, cate_df, cate_type_df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['%sday_user_action_1_ratio'%day] = 1.0*actions['%sday_user_action_4'%day] / actions['%sday_user_action_1'%day]
        actions['%sday_user_action_2_ratio'%day] = 1.0*actions['%sday_user_action_4'%day] / actions['%sday_user_action_2'%day]
        actions['%sday_user_action_3_ratio'%day] = 1.0*actions['%sday_user_action_4'%day] / actions['%sday_user_action_3'%day]
        actions['%sday_user_action_5_ratio'%day] = 1.0*actions['%sday_user_action_4'%day] / actions['%sday_user_action_5'%day]
        actions['%sday_user_action_6_ratio'%day] = 1.0*actions['%sday_user_action_4'%day] / actions['%sday_user_action_6'%day]

        actions = pd.merge(actions, lastbuy[['user_id','%sday_delta' % day]], how='left', on='user_id')
        actions = pd.merge(actions, active_num[['user_id','%sday_active_num' % day]], how='left', on='user_id')
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_labels(start_date, end_date):
    dump_path = path+'cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        label = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        candidates = actions[['user_id']].drop_duplicates().reset_index()

        actions = actions[(actions['type'] == 4)&(actions['cate']==8)]
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['label'] = 1
        label = pd.merge(candidates[['user_id']],actions[['user_id','label']],how='left',on='user_id')
        pickle.dump(actions, open(dump_path, 'w'))
    return label


def make_test_set(train_start_date, train_end_date):
    dump_path = path+'cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'r'))
    else:
        logging.info(u'用户基本特征...')
        user = get_basic_user_feat()

        # generate 时间窗口
        logging.info(u'基本特征集...')
        actions = get_basic_action_feat(train_start_date, train_end_date)
        #actions = None
        for i in (1,2, 3, 5, 7, 9, 10, 15,20, 30,40,50):
            logging.info(u'时间窗 %s days' % i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            #if actions is None:
                #actions = get_action_feat(start_days, train_end_date)
            #else:
            actions = pd.merge(actions, get_accumulate_user_feat(start_days, train_end_date,i), how='left',
                                   on=['user_id'])

        logging.info(u'聚合数据...')
        #actions = pd.merge(actions, labels, how='left', on='user_id')
        actions = pd.merge(actions, user, how='left', on='user_id')
        #actions = pd.merge(actions, user_acc, how='left', on='user_id')
        logging.info(u'数据聚合完毕...')
        #print actions.shape
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        
        #pickle.dump(actions,open(dump_path,'w'))      
    
    actions = actions.fillna(0)
    users = actions['user_id'].copy()
    del actions['user_id']
    
    return users, actions

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date):
    dump_path = path+'cache/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'r'))
    else:
        #start_days = "2016-03-05 00:00:00"
        logging.info(u'用户基本特征...')
        user = get_basic_user_feat()
        logging.info(u'用户标签数据...')
        labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        logging.info(u'基本特征集...')
        actions = get_basic_action_feat(train_start_date, train_end_date)
        #actions = None
        for i in (1,2, 3, 5, 7, 9, 10, 15,20, 30,40,50):
            logging.info(u'时间窗 %s days' % i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            #if actions is None:
                #actions = get_action_feat(start_days, train_end_date)
            #else:
            actions = pd.merge(actions, get_accumulate_user_feat(start_days, train_end_date,i), how='left',
                                   on=['user_id'])

        logging.info(u'聚合数据...')
        actions = pd.merge(actions, labels[['user_id','label']], how='left', on='user_id')
        actions = pd.merge(actions, user, how='left', on='user_id')
        #print actions.shape
        #actions = pd.merge(actions, user_acc, how='left', on='user_id')
        
        logging.info(u'数据聚合完毕...')
        #pickle.dump(actions,open(dump_path,'w'))
        
    #actions.to_csv('train.csv')
    actions = actions.fillna(0)
    labels = actions['label'].copy()
    users = actions['user_id'].copy()
    print u'正负样本比例 {0} : {1}'.format(actions[actions['label']==1].shape[0],actions[actions['label']==0].shape[0])  
    
    
    del actions['user_id']
    del actions['label']

    return users, actions, labels


def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)

def sub(train_start_date,train_end_date,pred):
    logging.info(u'开始生成推荐结果...')
    pred = pred.sort_values(by='label',ascending=False)[:20000]
    all_actions = get_actions(train_start_date, train_end_date)
    product = pd.read_csv(product_path)
    
    pred = pd.merge(pred, all_actions[['user_id','sku_id','type','time']].drop_duplicates(), how='left', on='user_id')
    result = pd.merge(pred, product[['sku_id','cate']], how='left', on='sku_id')
    result = result[result['cate']==8]

    print '过滤出观测日前加过购物车且没有下单和购物车删除的用户...'
    ifaddcart = result.groupby(['user_id','sku_id','label'], as_index=False).apply(lambda x: simple_choose(x))

    add_cart = ifaddcart[ifaddcart['add_cart']==1]
    add_cart = add_cart.sort_values(by='time',ascending=False).groupby(by='user_id').first()

    no_cart = ifaddcart[ifaddcart['add_cart']==0]
    no_cart = no_cart.sort_values(by=['label','time'],ascending=[0,0]).groupby(by='user_id').first()

    print u'结果生成...'
    pred = pd.concat([add_cart,no_cart[:12000-no_cart.shape[0]]])
    pred['sku_id'].to_csv(path+'sub/result.csv')
    return pred

if __name__=='__main__':
    sub('2016-02-06', '2016-04-16')


