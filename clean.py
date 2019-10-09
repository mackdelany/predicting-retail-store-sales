import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from features import lag_all_stores, add_lags_to_single_store, add_sales_per_customer, add_datetime_features_day_of_week, add_datetime_features_week

validation_sets = 3
max_train_size = 0.95

store = pd.read_csv('data/raw/store.csv')
train = pd.read_csv('data/raw/train.csv')

data = train[~train['Store'].isna()]
data = data.merge(store, how='left', left_on='Store', right_on='Store')

data = data.sort_values(by=['Date','Store'], ascending=['True','True']).reset_index(drop=True)

data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek

data['Customers'] = data.groupby(['Store','DayOfWeek'])['Customers'].transform(lambda x: x.fillna(x.mean()))
data['Sales'] = data.groupby(['Store','DayOfWeek'])['Sales'].transform(lambda x: x.fillna(x.mean()))

data = data.dropna(axis=1)
data = data.drop('DayOfWeek', axis=1)

data = add_datetime_features_day_of_week(data)
data = add_datetime_features_week(data)

data = data.drop('Date', axis=1)

tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)

fold=0

for train_index, test_index in tscv.split(data):
    
    fold_name = 'fold' + str(fold)
    
    data.iloc[train_index,:].drop('Sales',axis=1).to_csv('data/scenario_1_control/' + fold_name + '/' + fold_name + '_train_X.csv', index=False)
    data.iloc[train_index,:]['Sales'].to_csv('data/scenario_1_control/' + fold_name + '/' + fold_name + '_train_y.csv', index=False)
    data.iloc[test_index,:].drop('Sales',axis=1).to_csv('data/scenario_1_control/' + fold_name + '/' + fold_name + '_test_X.csv', index=False)
    data.iloc[test_index,:]['Sales'].to_csv('data/scenario_1_control/' + fold_name + '/' + fold_name + '_test_y.csv', index=False)

    data.iloc[train_index,:].drop('Sales',axis=1).to_csv('data/scenario_3_predicted_lags/' + fold_name + '/' + fold_name + '_train_X.csv', index=False)
    data.iloc[train_index,:]['Sales'].to_csv('data/scenario_3_predicted_lags/' + fold_name + '/' + fold_name + '_train_y.csv', index=False)
    data.iloc[test_index,:].drop('Sales',axis=1).to_csv('data/scenario_3_predicted_lags/' + fold_name + '/' + fold_name + '_test_X.csv', index=False)
    data.iloc[test_index,:]['Sales'].to_csv('data/scenario_3_predicted_lags/' + fold_name + '/' + fold_name + '_test_y.csv', index=False)
    
    fold += 1


lag_column = 'Sales'
lags=2

data = lag_all_stores(data, lag_column, lags)
data = data.dropna(subset=['Sales-lag-1','Sales-lag-2'], axis=0).reset_index(drop=True)

tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)

fold = 0

for train_index, test_index in tscv.split(data):
    
    fold_name = 'fold' + str(fold)
    
    data.iloc[train_index,:].drop('Sales',axis=1).to_csv('data/scenario_2_lags/' + fold_name + '/' + fold_name + '_train_X.csv', index=False)
    data.iloc[train_index,:]['Sales'].to_csv('data/scenario_2_lags/' + fold_name + '/' + fold_name + '_train_y.csv', index=False)
    data.iloc[test_index,:].drop('Sales',axis=1).to_csv('data/scenario_2_lags/' + fold_name + '/' + fold_name + '_test_X.csv', index=False)
    data.iloc[test_index,:]['Sales'].to_csv('data/scenario_2_lags/' + fold_name + '/' + fold_name + '_test_y.csv', index=False)
    
    fold += 1


print(data)