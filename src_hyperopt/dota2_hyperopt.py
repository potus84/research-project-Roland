
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
train = pd.read_csv("data/dota2/dota2.0.train", encoding='latin1', 
                 header=None,
                 na_values='?',
                 low_memory=False)

test = pd.read_csv("data/dota2/dota2.0.test", encoding='latin1', 
                 header=None,
                 na_values='?',
                 low_memory=False)
# In[8]:
y_train = train.iloc[:, 0]
X_train = train.iloc[:, 2:]


y_test = test.iloc[:, 0]
X_test = test.iloc[:, 2:]

# Standardize the output y
y_train[(y_train==-1)] = 0
y_test[(y_test==-1)] = 0


def objective(space):

    xgb = XGBClassifier(
        learning_rate =0.1,
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
            'n_estimators': hp.choice("n_estimators", range(350, 451, 10)),
            'max_depth': hp.choice("max_depth", range(1, 10)),
            'min_child_weight': hp.choice ('min_child_weight', range(65, 71)),
            'subsample': hp.uniform ('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform ('colsample_bytree', 0.6, 1),
            'gamma': hp.uniform ('gamma', 0, 0.5),
            'reg_alpha': hp.uniform ('reg_alpha', 0, 1e-2)
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
        learning_rate =0.1,        
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

