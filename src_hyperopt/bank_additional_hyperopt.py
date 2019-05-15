
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


os.chdir('/home/tai/research-project-Roland')


# ### Load train and test data

# In[8]:


train = pd.read_csv("data/bank_additional/bank_additional.0.train", encoding='latin1', 
                 names=['age',
                        'job',
                        'marital',
                        'education',
                        'default',
                        'housing',
                        'loan',
                        'contact',
                        'month',
                        'day_of_week',
                        'duration',
                        'campaign',
                        'pdays',
                        'previous',
                        'poutcome',
                        'emp_var_rate',
                        'cons_price_idx',
                        'cons_conf_idx',
                        'euribor3m',
                        'nr_employed',
                        'subscribe'],
                 na_values='?',
                 low_memory=False )


# In[9]:


test = pd.read_csv("data/bank_additional/bank_additional.0.test", encoding='latin1', 
                 names=['age',
                        'job',
                        'marital',
                        'education',
                        'default',
                        'housing',
                        'loan',
                        'contact',
                        'month',
                        'day_of_week',
                        'duration',
                        'campaign',
                        'pdays',
                        'previous',
                        'poutcome',
                        'emp_var_rate',
                        'cons_price_idx',
                        'cons_conf_idx',
                        'euribor3m',
                        'nr_employed',
                        'subscribe'],
                 na_values='?',
                 low_memory=False)


# ### Covert the output as binary

# In[11]:


train['has_subcribe'] = np.where(train.subscribe == 'yes', 1, 0)
train=train.drop(['subscribe'], axis=1)


# In[12]:


test['has_subcribe'] = np.where(test.subscribe == 'yes', 1, 0)
test=test.drop(['subscribe'], axis=1)


# ### Convert the numeric number

# In[13]:


train.loc[:,'age'] = pd.to_numeric(train['age'], downcast='integer', errors='coerce')
train.loc[:, 'duration'] = pd.to_numeric(train['duration'], downcast='integer', errors='coerce')
train.loc[:, 'campaign'] = pd.to_numeric(train['campaign'], downcast='integer', errors='coerce')
train.loc[:, 'pdays'] = pd.to_numeric(train['pdays'], downcast='integer', errors='coerce')
train.loc[:, 'previous'] = pd.to_numeric(train['previous'], downcast='integer', errors='coerce')
train.loc[:, 'emp_var_rate'] = pd.to_numeric(train['emp_var_rate'], downcast='float', errors='coerce')
train.loc[:, 'cons_price_idx'] = pd.to_numeric(train['cons_price_idx'], downcast='float', errors='coerce')
train.loc[:, 'cons_conf_idx'] = pd.to_numeric(train['cons_conf_idx'], downcast='float', errors='coerce')
train.loc[:, 'euribor3m'] = pd.to_numeric(train['euribor3m'], downcast='float', errors='coerce')
train.loc[:,'nr_employed'] = pd.to_numeric(train['nr_employed'], downcast='float', errors='coerce')


# In[14]:


test.loc[:,'age'] = pd.to_numeric(test['age'], downcast='integer', errors='coerce')
test.loc[:, 'duration'] = pd.to_numeric(test['duration'], downcast='integer', errors='coerce')
test.loc[:, 'campaign'] = pd.to_numeric(test['campaign'], downcast='integer', errors='coerce')
test.loc[:, 'pdays'] = pd.to_numeric(test['pdays'], downcast='integer', errors='coerce')
test.loc[:, 'previous'] = pd.to_numeric(test['previous'], downcast='integer', errors='coerce')
test.loc[:, 'emp_var_rate'] = pd.to_numeric(test['emp_var_rate'], downcast='float', errors='coerce')
test.loc[:, 'cons_price_idx'] = pd.to_numeric(test['cons_price_idx'], downcast='float', errors='coerce')
test.loc[:, 'cons_conf_idx'] = pd.to_numeric(test['cons_conf_idx'], downcast='float', errors='coerce')
test.loc[:, 'euribor3m'] = pd.to_numeric(test['euribor3m'], downcast='float', errors='coerce')
test.loc[:,'nr_employed'] = pd.to_numeric(test['nr_employed'], downcast='float', errors='coerce')


# ### One hot encoding

# In[18]:


train['job'] = train['job'].astype('category',
                                               categories=[
                                                 'admin',
                                                 'blue-collar',
                                                 'entrepreneur',
                                                 'housemaid',
                                                 'management',
                                                 'retired',
                                                 'self-employed',
                                                 'services',
                                                 'student',
                                                 'technician',
                                                 'unemployed'
                                               ])
train['marital'] = train['marital'].astype('category',
                                               categories=[
                                                 'divorced', 'married', 'single'
                                               ])

train['education'] = train['education'].astype('category',
                                               categories=[
                                                   'basic.4y',
                                                   'basic.6y',
                                                 'basic.9y',
                                                 'high.school',
                                                 'illiterate',
                                                 'professional.course',
                                                 'university.degree'])
train['default'] = train['default'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
train['housing'] = train['housing'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
train['loan'] = train['loan'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
train['contact'] = train['contact'].astype('category',
                                               categories=[
                                                 'cellular','telephone'
                                               ])
train['month'] = train['month'].astype('category',
                                               categories=[
                                                 'jan', 'feb', 'mar', 
                                                   'apr', 'may', 'jun', 
                                                   'jul', 'aug', 'sep',
                                                   'oct', 'nov', 'dec'
                                               ])
train['day_of_week'] = train['day_of_week'].astype('category',
                                               categories=[
                                                 'mon','tue','wed','thu','fri'
                                               ])
train['poutcome'] = train['poutcome'].astype('category',
                                               categories=[
                                                'failure','nonexistent','success'
                                               ])
                                                   


# In[19]:


train = pd.get_dummies(train, columns=['contact', 'month', 'day_of_week', 'poutcome'])
train = pd.get_dummies(train, columns=['job', 'marital', 'education', 'default', 'housing', 'loan'], dummy_na=True)


# In[20]:


test['job'] = test['job'].astype('category',
                                               categories=[
                                                 'admin',
                                                 'blue-collar',
                                                 'entrepreneur',
                                                 'housemaid',
                                                 'management',
                                                 'retired',
                                                 'self-employed',
                                                 'services',
                                                 'student',
                                                 'technician',
                                                 'unemployed'
                                               ])
test['marital'] = test['marital'].astype('category',
                                               categories=[
                                                 'divorced', 'married', 'single'
                                               ])

test['education'] = test['education'].astype('category',
                                               categories=[
                                                   'basic.4y',
                                                   'basic.6y',
                                                 'basic.9y',
                                                 'high.school',
                                                 'illiterate',
                                                 'professional.course',
                                                 'university.degree'])
test['default'] = test['default'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
test['housing'] = test['housing'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
test['loan'] = test['loan'].astype('category',
                                               categories=[
                                                 'no', 'yes'
                                               ])
test['contact'] = test['contact'].astype('category',
                                               categories=[
                                                 'cellular','telephone'
                                               ])
test['month'] = test['month'].astype('category',
                                               categories=[
                                                 'jan', 'feb', 'mar', 
                                                   'apr', 'may', 'jun', 
                                                   'jul', 'aug', 'sep',
                                                   'oct', 'nov', 'dec'
                                               ])
test['day_of_week'] = test['day_of_week'].astype('category',
                                               categories=[
                                                 'mon','tue','wed','thu','fri'
                                               ])
test['poutcome'] = test['poutcome'].astype('category',
                                               categories=[
                                                'failure','nonexistent','success'
                                               ])
                                                   


# In[21]:


test = pd.get_dummies(test, columns=['contact', 'month', 'day_of_week', 'poutcome'])
test = pd.get_dummies(test, columns=['job', 'marital', 'education', 'default', 'housing', 'loan'], dummy_na=True)


# In[22]:


X_train = train.drop(['has_subcribe'], axis=1)
y_train = train.has_subcribe

X_test = test.drop(['has_subcribe'], axis=1)
y_test = test.has_subcribe


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
            'n_estimators': hp.choice("n_estimators", range(750, 801, 10)),
            'max_depth': hp.choice("max_depth", range(1,10,1)),
            'min_child_weight': hp.choice ('min_child_weight', range(1,10,1)),
            'subsample': hp.uniform ('subsample', 0.7, 1),
            'colsample_bytree': hp.uniform ('colsample_bytree', 0.7, 1),
            'gamma': hp.uniform ('gamma', 0.1, 0.5),
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

