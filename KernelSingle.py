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
# this class define various trainning methods, also cross validation to tune the hyper parameters
class Train:

  # The constructor takes a pandas dataframe as input and save it to self.df
  def __init__(self, df_train):
    self.df = df_train

  # This method evaluate the preprocessing with rf regression model, calculate the mean absolute error. Be careful! this method can only used in train set!
  def PreprocessingScore(self):
    X_train, X_test, y_train, y_test = train_test_split(
                                                        self.df.drop(['id', 'target'],axis=1),
                                                        self.df.target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0
                                                       )

    sc_mod = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', max_depth=10, min_samples_split=15, min_samples_leaf=10, n_jobs=-1, random_state=0)
    sc_mod.fit(X_train, y_train)
    ysc_pred = sc_mod.predict_proba(X_test)[:,1]
    #print (ysc_pred)
    #print (y_test)
    gini = 2*roc_auc_score(y_test, ysc_pred)-1
    return gini

  def rf_param_selection(self, nfolds):
    sc_mod = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=0, verbose=True)
    #nests = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    mfeat = [6, 7, 8, 9, 10, 11]
    nests = [1000]
    #mfeat = [6,7]
    #mdeps = [8, 9, 10, 11, 12, 13, 14]
    param_grid = {'n_estimators': nests, 'max_features' : mfeat}
    grid_search = GridSearchCV(sc_mod, param_grid, scoring='roc_auc', cv=nfolds, verbose=2)
    grid_search.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    #print ("CV results")
    #print (grid_search.cv_results_)
    print ("Grid Scores:")
    print (grid_search.grid_scores_)
    print ("Best parameters:")
    print (grid_search.best_params_)
    return grid_search.cv_results_

  # A simple train of random forest model with scikit learn
  def TrainSKLearnRandomForest(self):
    rf = RandomForestClassifier(n_estimators=500, max_features='auto', criterion='gini', max_depth=10, min_samples_leaf=10, min_samples_split=15, n_jobs=-1, random_state=0)
    rf.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    #features = self.df.drop(['id', 'target'],axis=1).columns.values
    return rf

  # XGBoost
  def TrainXGBoost(self):
    xgb = XGBClassifier(    
                        n_estimators=1000,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=.77,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                       )
    fit_xgb = xgb.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    return fit_xgb
 
  # XGBoost CV
  def xgb_cv(self):
    xgbcvtrainDM = xgboost.DMatrix(self.df.drop(['id', 'target'],axis=1), label=self.df.target)
    xgb_param = { 'n_estimators':1000, 'max_depth':4, 'eta':0.07, 'silent':1, 'objective':'binary:logistic' }
    #print('running cross validation, XGB')
    #xgboost.cv( xgb_param, xgbcvtrainDM, 2, nfold=5, metrics={'error'}, seed=0, callbacks=[xgboost.callback.print_evaluation(show_stdv=True)] )
    print('running cross validation, XGB, disable standard deviation display')
    res = xgboost.cv( xgb_param, xgbcvtrainDM, num_boost_round=5000, nfold=5, metrics={'auc'}, seed=0, callbacks=[xgboost.callback.print_evaluation(show_stdv=False), xgboost.callback.early_stop(4)] )
    print(res)

  # light gbm
  def TrainLightGBM(self):

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

    lgb_params_2 = {}
    lgb_params_2['n_estimators'] = 1090
    lgb_params_2['learning_rate'] = 0.02   
    lgb_params_2['subsample'] = 0.7
    lgb_params_2['subsample_freq'] = 2
    lgb_params_2['num_leaves'] = 16
    lgb_params_2['feature_fraction'] = 0.8
    lgb_params_2['bagging_freq'] = 1
    lgb_params_2['random_state'] = 20

    lgb_params_3 = {}
    lgb_params_3['n_estimators'] = 1100
    lgb_params_3['max_depth'] = 4
    lgb_params_3['learning_rate'] = 0.02
    lgb_params_3['feature_fraction'] = 0.95
    lgb_params_3['bagging_freq'] = 1
    lgb_params_3['random_state'] = 25
    
    lgb_model_1 = LGBMClassifier(**lgb_params_1)
    #lgb_model_2 = LGBMClassifier(**lgb_params_2)
    #lgb_model_3 = LGBMClassifier(**lgb_params_3)
    
    #lgb_model.fit(X_train, y_train)
    #fit_lgb = lgb_model.fit(self.df.drop(['id', 'target'],axis=1), self.df.target)
    #return fit_lgb

    scores = cross_val_score(lgb_model_1, self.df.drop(['id', 'target'],axis=1), self.df.target, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
    #scores = cross_val_score(lgb_model_2, self.df.drop(['id', 'target'],axis=1), self.df.target, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
    #scores = cross_val_score(lgb_model_3, self.df.drop(['id', 'target'],axis=1), self.df.target, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
    print(scores)
    return scores

# this class makes prediction, with various methods
class Prediction:

  # The constructor takes model and test data for 
  def __init__(self, df_test, thismodel):
    self.df = df_test
    self.model = thismodel

  # prediction with simple random forest model
  def PredSKLearnRandomForest(self):
    pred = self.model.predict_proba(self.df.drop(['id'],axis=1))
    return pred[:,1]

  # prediction with xgb
  def PredXGBoost(self):
    pred = self.model.predict_proba(self.df.drop(['id'],axis=1))[:,1]
    return pred

  # prediction with light gbm
  def PredLightGBM(self):
    pred = self.model.predict_proba(self.df.drop(['id'],axis=1))
    return pred

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
  
  train = Train(train_p)
  #print (train.PreprocessingScore())
  #scan_res = train.rf_param_selection(5)
  #train.xgb_cv()
  train.TrainLightGBM()
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

