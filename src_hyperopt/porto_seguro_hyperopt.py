
# coding: utf-8

# # Bank additional dataset

# ## Part 1: Data encoding

# ### Import library

# In[6]:



import pandas as pd
import numpy as np
import os
import xgboost

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval


# In[7]:


os.chdir('/home/tai/Projects/research-project-Roland')


# ### Load train and test data

# In[8]:

train = pd.read_csv("data/porto_seguro/porto_seguro.0.train", encoding='latin1',
                 na_values='?',
                 names=['id',
                 'target',
                 'ps_ind_01',
                 'ps_ind_02_cat',
                 'ps_ind_03',
                 'ps_ind_04_cat',
                 'ps_ind_05_cat',
                 'ps_ind_06_bin',
                 'ps_ind_07_bin',
                 'ps_ind_08_bin',
                 'ps_ind_09_bin',
                 'ps_ind_10_bin',
                 'ps_ind_11_bin',
                 'ps_ind_12_bin',
                 'ps_ind_13_bin',
                 'ps_ind_14',
                 'ps_ind_15',
                 'ps_ind_16_bin',
                 'ps_ind_17_bin',
                 'ps_ind_18_bin',
                 'ps_reg_01',
                 'ps_reg_02',
                 'ps_reg_03',
                 'ps_car_01_cat',
                 'ps_car_02_cat',
                 'ps_car_03_cat',
                 'ps_car_04_cat',
                 'ps_car_05_cat',
                 'ps_car_06_cat',
                 'ps_car_07_cat',
                 'ps_car_08_cat',
                 'ps_car_09_cat',
                 'ps_car_10_cat',
                 'ps_car_11_cat',
                 'ps_car_11',
                 'ps_car_12',
                 'ps_car_13',
                 'ps_car_14',
                 'ps_car_15',
                 'ps_calc_01',
                 'ps_calc_02',
                 'ps_calc_03',
                 'ps_calc_04',
                 'ps_calc_05',
                 'ps_calc_06',
                 'ps_calc_07',
                 'ps_calc_08',
                 'ps_calc_09',
                 'ps_calc_10',
                 'ps_calc_11',
                 'ps_calc_12',
                 'ps_calc_13',
                 'ps_calc_14',
                 'ps_calc_15_bin',
                 'ps_calc_16_bin',
                 'ps_calc_17_bin',
                 'ps_calc_18_bin',
                 'ps_calc_19_bin',
                 'ps_calc_20_bin'],
                 low_memory=False)


# In[9]:


test = pd.read_csv("data/porto_seguro/porto_seguro.0.test", encoding='latin1', 
                 names=['id',
                 'target',
                 'ps_ind_01',
                 'ps_ind_02_cat',
                 'ps_ind_03',
                 'ps_ind_04_cat',
                 'ps_ind_05_cat',
                 'ps_ind_06_bin',
                 'ps_ind_07_bin',
                 'ps_ind_08_bin',
                 'ps_ind_09_bin',
                 'ps_ind_10_bin',
                 'ps_ind_11_bin',
                 'ps_ind_12_bin',
                 'ps_ind_13_bin',
                 'ps_ind_14',
                 'ps_ind_15',
                 'ps_ind_16_bin',
                 'ps_ind_17_bin',
                 'ps_ind_18_bin',
                 'ps_reg_01',
                 'ps_reg_02',
                 'ps_reg_03',
                 'ps_car_01_cat',
                 'ps_car_02_cat',
                 'ps_car_03_cat',
                 'ps_car_04_cat',
                 'ps_car_05_cat',
                 'ps_car_06_cat',
                 'ps_car_07_cat',
                 'ps_car_08_cat',
                 'ps_car_09_cat',
                 'ps_car_10_cat',
                 'ps_car_11_cat',
                 'ps_car_11',
                 'ps_car_12',
                 'ps_car_13',
                 'ps_car_14',
                 'ps_car_15',
                 'ps_calc_01',
                 'ps_calc_02',
                 'ps_calc_03',
                 'ps_calc_04',
                 'ps_calc_05',
                 'ps_calc_06',
                 'ps_calc_07',
                 'ps_calc_08',
                 'ps_calc_09',
                 'ps_calc_10',
                 'ps_calc_11',
                 'ps_calc_12',
                 'ps_calc_13',
                 'ps_calc_14',
                 'ps_calc_15_bin',
                 'ps_calc_16_bin',
                 'ps_calc_17_bin',
                 'ps_calc_18_bin',
                 'ps_calc_19_bin',
                 'ps_calc_20_bin'],
                 na_values='?',
                 low_memory=False)


X_train = train.drop(['target', 'id'], axis=1)
y_train = train.target

X_test = test.drop(['target', 'id'], axis=1)
y_test = test.target


def objective(space):

    xgb = XGBClassifier(
        learning_rate =0.01,
        n_estimators=space["n_estimators"],
        max_depth=space['max_depth'],
        min_child_weight=space['min_child_weight'],
        reg_alpha=space['reg_alpha'],
        gamma=space['gamma'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        objective= 'binary:logistic',
        n_jobs=-1
    )
    
    kfold = StratifiedKFold(n_splits=5, random_state=0)
    results = cross_val_score(xgb, X_train, y_train, cv=kfold)


     
    test_mean_acc = results.mean()
    
    return{'loss':-test_mean_acc, 'status': STATUS_OK }

space = {
            'n_estimators': hp.choice("n_estimators", range(650, 751, 10)),
            'max_depth': hp.choice("max_depth", range(1,10,1)),
            'min_child_weight': hp.choice ('min_child_weight', range(150, 171)),
            'subsample': hp.uniform ('subsample', 0.7, 1),
            'colsample_bytree': hp.uniform ('colsample_bytree', 0.7, 1),
            'gamma': hp.uniform ('gamma', 0.1, 0.5),
            'reg_alpha': hp.uniform ('reg_alpha', 5e-5, 5e-4)
        }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

best_params = space_eval(space, best)
print (best_params)
# Test on the test set
NUM_TRIALS = int(np.ceil(200000/train.shape[0]))
print ('Test {} times on the test set'.format(NUM_TRIALS))
accuracy_array = []
for i in range(NUM_TRIALS):
    xgb = XGBClassifier(
        **best_params,
        learning_rate =0.01,        
        objective= 'binary:logistic',
        n_jobs=-1,
        scale_pos_weight=1,
        seed=i
    )
    model = xgb.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_array.append(accuracy)
    print('Accuracy {}: %.2f%%'.format(i) % (accuracy * 100.0))
mean_accuracy_score = sum(accuracy_array) / NUM_TRIALS
print('Average accuracy is: %.2f%%' % (mean_accuracy_score * 100.0))

