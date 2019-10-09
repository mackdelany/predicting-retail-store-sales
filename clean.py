import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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

tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)

fold=0

for train_index, test_index in tscv.split(data):
    
    fold_name = 'fold' + str(fold)
    
    data.iloc[train_index,:].drop('Sales',axis=1).to_csv('data/' + fold_name + '/' + fold_name + '_train_X.csv')
    data.iloc[train_index,:]['Sales'].to_csv('data/' + fold_name + '/' + fold_name + '_train_y.csv')
    data.iloc[test_index,:].drop('Sales',axis=1).to_csv('data/' + fold_name + '/' + fold_name + '_test_X.csv')
    data.iloc[test_index,:]['Sales'].to_csv('data/' + fold_name + '/' + fold_name + '_test_y.csv')
    
    fold += 1