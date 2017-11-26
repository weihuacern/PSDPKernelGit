import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
# This class is for the preprocessing, take csv path as input, and aim to return a pandas data frame for trainning
class PreProcessing:

  # The constructor takes a pandas dataframe as input and save it to self.df
  def __init__(self, csvpath):
    self.df = pd.read_csv(csvpath)

  # This method have deal with missing data before merge or drop
  def MissingData(self):
    self.df = self.df.replace(-1, np.NaN)
    #print (self.df.columns[self.df.isnull().any()])
    '''
    'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 
    'ps_reg_03',
    'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat', 
    'ps_car_11', 'ps_car_12', 'ps_car_14'
    '''
    mean_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    mdan_imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    mfrq_imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    self.df["ps_ind_02_cat"].fillna(-1, inplace=True)
    self.df["ps_ind_04_cat"].fillna(-1, inplace=True)
    self.df["ps_ind_05_cat"].fillna(-1, inplace=True)
    #self.df["ps_reg_03"].fillna(self.df["ps_reg_03"].median(), inplace=True)
    self.df['ps_reg_03'] = mean_imp.fit_transform(self.df[['ps_reg_03']]).ravel()
    self.df["ps_car_01_cat"].fillna(-1, inplace=True)
    self.df["ps_car_02_cat"].fillna(-1, inplace=True)
    self.df["ps_car_03_cat"].fillna(self.df["ps_car_03_cat"].value_counts().idxmax(), inplace=True) # top 1 missing variable, drop
    self.df["ps_car_05_cat"].fillna(self.df["ps_car_05_cat"].value_counts().idxmax(), inplace=True) # top 2 missing variable, drop
    self.df["ps_car_07_cat"].fillna(-1, inplace=True)
    self.df["ps_car_09_cat"].fillna(-1, inplace=True)
    #self.df["ps_car_11"].fillna(self.df["ps_car_11"].value_counts().idxmax(), inplace=True)
    self.df['ps_car_11'] = mfrq_imp.fit_transform(self.df[['ps_car_11']]).ravel()
    #self.df["ps_car_12"].fillna(self.df["ps_car_12"].median(), inplace=True)
    self.df['ps_car_12'] = mean_imp.fit_transform(self.df[['ps_car_12']]).ravel()
    #self.df["ps_car_14"].fillna(self.df["ps_car_14"].median(), inplace=True)
    self.df['ps_car_14'] = mean_imp.fit_transform(self.df[['ps_car_14']]).ravel()
    #self.df[""].fillna(self.df[""].mean(), inplace=True)
    #self.df[""].fillna(self.df[""].median(), inplace=True)
    #self.df[""].fillna(self.df[""].value_counts().idxmax(), inplace=True)
    return

  # This method drop or merge variables in dataframe accroding to corr map
  def CorrMergeDrop(self):
    self.df['ps_ind_06070809_bin'] = self.df.apply(
      lambda x: 1 if x['ps_ind_06_bin'] == 1 
                  else 
                  (2 if x['ps_ind_07_bin'] == 1 
                     else 
                     ( 3 if x['ps_ind_08_bin'] == 1 
                         else 
                         (4 if x['ps_ind_09_bin'] == 1 else 5)
                     )
                  ), axis = 1)
    self.df.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

    self.df['ps_ind_161718_bin'] = self.df.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 
                                                             else (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1)
    self.df.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)
 
    # drop this variable from preprocessing study, top 3 missing data, and not important at all
    self.df.drop(['ps_car_03_cat'], axis = 1, inplace = True)
    self.df.drop(['ps_car_05_cat'], axis = 1, inplace = True)
   
    #self.df['ps_car_13'] = (self.df['ps_car_13']*self.df['ps_car_13']*48400).round(0)
    #self.df['ps_car_12'] = (self.df['ps_car_12']*self.df['ps_car_12']).round(4) * 10000
    return

  def ScaleFeatures(self):
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(self.df.drop(['id','target'], axis=1))

  # this method pack all previous preprocessing all together and return the data frame
  def FinalFrameforTrainning(self):
    self.MissingData()
    self.CorrMergeDrop()
    self.ScaleFeatures()
    return self.df

from sklearn.metrics import roc_auc_score
# this is a simple function to calculate gini score, for cross validation
def GiniScore(y_actual, y_pred):
  return 2*roc_auc_score(y_actual, y_pred)-1

# Class to extend the Sklearn classifier
class SklearnHelper(object):
  def __init__(self, clf, seed=0, params=None):
    params['random_state'] = seed
    self.clf = clf(**params)

  def train(self, x_train, y_train):
    self.clf.fit(x_train, y_train)

  def predict(self, x):
    return self.clf.predict(x)
    
  def fit(self,x,y):
    return self.clf.fit(x,y)
    
  def feature_importances(self,x,y):
    print(self.clf.fit(x,y).feature_importances_)

# Create submission file
class CreateSubmitCSV:

  # The constructor ypred and test dataframe as input 
  def __init__(self, test_id, pred_target):
    self.testid = test_id
    self.predtarget = pred_target

  def Create(self):
    # Create submission file
    sub = pd.DataFrame()
    sub['id'] = self.testid
    sub['target'] = self.predtarget
    sub.to_csv('this_submit.csv', index=False)

if __name__ == '__main__':
  #'''
  preprocessing = PreProcessing('../data/train.csv')
  #preprocessing = PreProcessing('smalltrain.csv')
  train_p = preprocessing.FinalFrameforTrainning()
  print ("done with trainning set preprocessing!")
  #train_p.to_csv('train_p.csv', index = False)
  #preprocessing = PreProcessing('data/test.csv')
  #test_p = preprocessing.FinalFrameforTrainning()
  #print ("done with test set preprocessing!")
  #test_p.to_csv('test_p.csv', index = False)
  #train_p = pd.read_csv('train_p.csv')
  #test_p = pd.read_csv('test_p.csv')
 
  # Some useful parameters which will come in handy later on
  #ntrain = train.shape[0]
  #ntest = test.shape[0]
  SEED = 0 # for reproducibility
  NFOLDS = 5 # set folds for out-of-fold prediction
  #kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
  # Load models
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import mean_absolute_error
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import cross_val_score
  from sklearn import metrics
  from xgboost import XGBClassifier
  import xgboost
  from lightgbm import LGBMClassifier
  import lightgbm

  # Put in our parameters for said classifiers
  # Random Forest parameters
  rf_params = 
  {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True, 
    #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 2
  }

  # Extreme Gradient Boosting parameters
  xgb_params = 
  {
    'n_estimators': 1000,
    'max_depth': 4,
    'objective': "binary:logistic",
    'learning_rate': 0.07,
    'subsample': 0.8,
    'min_child_weight': 0.77,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.6,
    'gamma': 10,
    'reg_alpha': 8,
    'reg_lambda': 1.3,
    'verbose': 2
  }

  lgb_params_1 = {}
  lgb_params_1['n_estimators'] = 650
  lgb_params_1['learning_rate'] = 0.02
  lgb_params_1['max_bin'] = 10
  lgb_params_1['subsample'] = 0.8
  #lgb_params_1['subsample_freq'] = 10  
  lgb_params_1['min_child_samples'] = 500
  lgb_params_1['feature_fraction'] = 0.9
  lgb_params_1['bagging_freq'] = 1
  lgb_params_1['random_state'] = 15

  rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
  xgb = SklearnHelper(clf=XGBClassifier, seed=SEED, params=xgb_params)
  lgb_1 = SklearnHelper(clf=LGBMClassifier, seed=SEED, params=lgb_params_1)
  #train = Train(train_p)
  #print (train.PreprocessingScore())
  #scan_res = train.rf_param_selection(5)
  #train.xgb_cv()
  #train.TrainLightGBM()
  #j = json.dumps(scan_res, indent=2)
  #f = open('sample.json', 'w')
  #print >> f, j
  #f.close()
  #np.savetxt('PgridScan.txt', train.rf_param_selection(2))
  '''
  thisrf = train.TrainSKLearnRandomForest()
  #thisxgb = train.TrainXGBoost()
  print ("done with model trainning!")
  
  prediction = Prediction(test_p, thisrf)
  ypred = prediction.PredSKLearnRandomForest()
  #prediction = Prediction(test_p, thisxgb)
  #ypred = prediction.PredXGBoost()
  print ("done with prediction!")
  print (ypred) 

  outcsv = CreateSubmitCSV(test_p['id'], ypred)
  outcsv.Create()
  print ("done with csv output!")
  '''

