import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

validation_sets = 3
max_train_size = 0.95

store = pd.read_csv('data/raw/store.csv')
train = pd.read_csv('data/raw/train.csv')

data = train[~train['Store'].isna()]
data = data.merge(store, how='left', left_on='Store', right_on='Store')





tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)

fold=0

for train_index, test_index in tscv.split(data):
    
    fold_name = 'fold' + str(fold)
    
    data.iloc[train_index,:].to_csv('data/' + fold_name + '/' + fold_name + '_train.csv')
    data.iloc[test_index,:].to_csv('data/' + fold_name + '/' + fold_name + '_test.csv')
    
    fold += 1