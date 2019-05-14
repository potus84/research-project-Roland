
# coding: utf-8

# # Adult dataset

# ## Part 1: Data encoding

# ### Import library

# In[1]:


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

# In[2]:


os.chdir('/home/tai/Projects/research-project-Roland/')


# ### Load train and test data

# In[3]:


train = pd.read_csv("data/adult/adult.0.train.csv", encoding='latin1', 
                 names=['age','workclass','fnlwgt','education',
                         'education_num','marital_status','occupation',
                         'relationship','race','sex','capital_gain','capital_loss',
                         'hours_per_week','native_country','income'],
                 na_values='?',
                 low_memory=False )


# In[4]:


test = pd.read_csv("data/adult/adult.0.test.csv", encoding='latin1', 
                 names=['age','workclass','fnlwgt','education',
                         'education_num','marital_status','occupation',
                         'relationship','race','sex','capital_gain','capital_loss',
                         'hours_per_week','native_country','income'],
                 na_values='?',
                 low_memory=False)


# ### Covert the output as binary

# In[5]:


train['over_50k'] = np.where(train.income == '>50K', 1, 0)
train=train.drop(['income'], axis=1)


# In[6]:


test['over_50k'] = np.where(test.income == '>50K', 1, 0)
test=test.drop(['income'], axis=1)


# ### Convert the numeric number

# In[7]:


train.loc[:,'age'] = pd.to_numeric(train['age'], downcast='integer', errors='coerce')
train.loc[:,'fnlwgt'] = pd.to_numeric(train['fnlwgt'], downcast='float', errors='coerce')
train.loc[:,'age'] = pd.to_numeric(train['age'], downcast='integer', errors='coerce')
train.loc[:,'capital_gain'] = pd.to_numeric(train['capital_gain'], downcast='float', errors='coerce')
train.loc[:,'capital_loss'] = pd.to_numeric(train['capital_loss'], downcast='float', errors='coerce')
train.loc[:,'hours_per_week'] = pd.to_numeric(train['hours_per_week'], downcast='float', errors='coerce')


# In[8]:


test.loc[:,'age'] = pd.to_numeric(test['age'], downcast='integer', errors='coerce')
test.loc[:,'fnlwgt'] = pd.to_numeric(test['fnlwgt'], downcast='float', errors='coerce')
test.loc[:,'age'] = pd.to_numeric(test['age'], downcast='integer', errors='coerce')
test.loc[:,'capital_gain'] = pd.to_numeric(test['capital_gain'], downcast='float', errors='coerce')
test.loc[:,'capital_loss'] = pd.to_numeric(test['capital_loss'], downcast='float', errors='coerce')
test.loc[:,'hours_per_week'] = pd.to_numeric(test['hours_per_week'], downcast='float', errors='coerce')


# ### One hot encoding

# In[11]:


train['education'] = train['education'].astype('category',
                                               categories=['Bachelors', 'Some-college', '11th', 'HS-grad', 
                                                           'Prof-school', 
                                                           'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
                                                           '12th', 'Masters', '1st-4th', '10th', 
                                                           'Doctorate', '5th-6th', 'Preschool'])
train['marital_status'] = train['marital_status'].astype('category',
                                                         categories=['Married-civ-spouse', 'Divorced', 
                                                                     'Never-married', 'Separated', 
                                                                     'Widowed', 'Married-spouse-absent', 
                                                                     'Married-AF-spouse'])
train['relationship'] = train['relationship'].astype('category',
                                                     categories=['Wife', 'Own-child', 'Husband', 
                                                                 'Not-in-family', 'Other-relative', 'Unmarried'])
train['race'] = train['race'].astype('category',
                                     categories=['White', 'Asian-Pac-Islander', 
                                                 'Amer-Indian-Eskimo', 'Other', 'Black'])
train['sex'] = train['sex'].astype('category', 
                                   categories=['Female', 'Male'])


train['workclass'] = train['workclass'].astype('category',
                                               categories=['Private', 'Self-emp-not-inc', 
                                                           'Self-emp-inc', 'Federal-gov', 
                                                           'Local-gov', 'State-gov', 
                                                           'Without-pay', 'Never-worked'])
train['occupation'] = train['occupation'].astype('category',
                                                 categories=['Tech-support', 'Craft-repair', 
                                                             'Other-service', 'Sales', 'Exec-managerial',
                                                             'Prof-specialty', 'Handlers-cleaners', 
                                                             'Machine-op-inspct', 'Adm-clerical',
                                                             'Farming-fishing', 'Transport-moving', 
                                                             'Priv-house-serv',
                                                             'Protective-serv', 'Armed-Forces'])
train['native_country'] = train['native_country'].astype('category',
                                                         categories=['United-States',
                                                                                 'Cambodia',
                                                                                 'England',
                                                                                 'Puerto-Rico',
                                                                                 'Canada',
                                                                                 'Germany',
                                                                                 'Outlying-US(Guam-USVI-etc)',
                                                                                 'India',
                                                                                 'Japan',
                                                                                 'Greece',
                                                                                 'South',
                                                                                 'China',
                                                                                 'Cuba',
                                                                                 'Iran',
                                                                                 'Honduras',
                                                                                 'Philippines',
                                                                                 'Italy',
                                                                                 'Poland',
                                                                                 'Jamaica',
                                                                                 'Vietnam',
                                                                                 'Mexico',
                                                                                 'Portugal',
                                                                                 'Ireland',
                                                                                 'France',
                                                                                 'Dominican-Republic',
                                                                                 'Laos',
                                                                                 'Ecuador',
                                                                                 'Taiwan',
                                                                                 'Haiti',
                                                                                 'Columbia',
                                                                                 'Hungary',
                                                                                 'Guatemala',
                                                                                 'Nicaragua',
                                                                                 'Scotland',
                                                                                 'Thailand',
                                                                                 'Yugoslavia',
                                                                                 'El-Salvador',
                                                                                 'Trinadad&Tobago',
                                                                                 'Peru',
                                                                                 'Hong',
                                                                                 'Holand-Netherlands'])


# In[12]:


train = pd.get_dummies(train, columns=['education','marital_status','relationship','race','sex'])
train = pd.get_dummies(train, columns=['workclass','occupation','native_country'], dummy_na=True)


# In[13]:


test['education'] = test['education'].astype('category',
                                               categories=['Bachelors', 'Some-college', '11th', 'HS-grad', 
                                                           'Prof-school', 
                                                           'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th',
                                                           '12th', 'Masters', '1st-4th', '10th', 
                                                           'Doctorate', '5th-6th', 'Preschool'])
test['marital_status'] = test['marital_status'].astype('category',
                                                         categories=['Married-civ-spouse', 'Divorced', 
                                                                     'Never-married', 'Separated', 
                                                                     'Widowed', 'Married-spouse-absent', 
                                                                     'Married-AF-spouse'])
test['relationship'] = test['relationship'].astype('category',
                                                     categories=['Wife', 'Own-child', 'Husband', 
                                                                 'Not-in-family', 'Other-relative', 'Unmarried'])
test['race'] = test['race'].astype('category',
                                    categories=['White', 'Asian-Pac-Islander', 
                                                 'Amer-Indian-Eskimo', 'Other', 'Black'])
test['sex'] = test['sex'].astype('category',
                                   categories=['Female', 'Male'])


test['workclass'] = test['workclass'].astype('category',
                                               categories=['Private', 'Self-emp-not-inc', 
                                                           'Self-emp-inc', 'Federal-gov', 
                                                           'Local-gov', 'State-gov', 
                                                           'Without-pay', 'Never-worked'])
test['occupation'] = test['occupation'].astype('category',
                                                categories=['Tech-support', 'Craft-repair', 
                                                             'Other-service', 'Sales', 'Exec-managerial',
                                                             'Prof-specialty', 'Handlers-cleaners', 
                                                             'Machine-op-inspct', 'Adm-clerical',
                                                             'Farming-fishing', 'Transport-moving', 
                                                             'Priv-house-serv',
                                                             'Protective-serv', 'Armed-Forces'])
test['native_country'] = test['native_country'].astype('category',                                                         
                                                         categories=['United-States',
                                                                                 'Cambodia',
                                                                                 'England',
                                                                                 'Puerto-Rico',
                                                                                 'Canada',
                                                                                 'Germany',
                                                                                 'Outlying-US(Guam-USVI-etc)',
                                                                                 'India',
                                                                                 'Japan',
                                                                                 'Greece',
                                                                                 'South',
                                                                                 'China',
                                                                                 'Cuba',
                                                                                 'Iran',
                                                                                 'Honduras',
                                                                                 'Philippines',
                                                                                 'Italy',
                                                                                 'Poland',
                                                                                 'Jamaica',
                                                                                 'Vietnam',
                                                                                 'Mexico',
                                                                                 'Portugal',
                                                                                 'Ireland',
                                                                                 'France',
                                                                                 'Dominican-Republic',
                                                                                 'Laos',
                                                                                 'Ecuador',
                                                                                 'Taiwan',
                                                                                 'Haiti',
                                                                                 'Columbia',
                                                                                 'Hungary',
                                                                                 'Guatemala',
                                                                                 'Nicaragua',
                                                                                 'Scotland',
                                                                                 'Thailand',
                                                                                 'Yugoslavia',
                                                                                 'El-Salvador',
                                                                                 'Trinadad&Tobago',
                                                                                 'Peru',
                                                                                 'Hong',
                                                                                 'Holand-Netherlands'])


# In[14]:


test = pd.get_dummies(test, columns=['education','marital_status','relationship','race','sex'])
test = pd.get_dummies(test, columns=['workclass','occupation','native_country'], dummy_na=True)


# In[15]:


X_train = train.drop(['over_50k'], axis=1)
y_train = train.over_50k

X_test = test.drop(['over_50k'], axis=1)
y_test = test.over_50k


# ## Part 2: Tuning on train data

# ## Tuning using Hyperopt

# In[23]:


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
            'n_estimators': hp.choice("n_estimators", range(1500, 2501, 100)),
            'max_depth': hp.choice("max_depth", range(1,10,1)),
            'min_child_weight': hp.choice ('min_child_weight', range(1,10,1)),
            'subsample': hp.uniform ('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform ('colsample_bytree', 0.6, 1),
            'gamma': hp.uniform ('gamma', 0.1, 0.5),
            'reg_alpha': hp.uniform ('reg_alpha', 1e-2, 0.1)
        }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1,
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


