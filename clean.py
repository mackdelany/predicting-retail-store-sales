import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from features import lag_all_stores, add_lags_to_single_store, add_sales_per_customer, add_datetime_features_day_of_week, add_datetime_features_week


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='local', nargs='?')
    parser.add_argument('--cpu', default=8, nargs='?')
    args = parser.parse_args()

    validation_sets = 3
    max_train_size = 0.95

    store = pd.read_csv('data/raw/store.csv')
    train = pd.read_csv('data/raw/train.csv')
    print('train shape {}'.format(train.shape))

    #  does this do anything?
    data = train[~train['Store'].isna()]
    print('train shape {}'.format(train.shape))

    #  why merge with stores here?
    data = data.merge(store, how='left', left_on='Store', right_on='Store')
    data = data.sort_values(by=['Date','Store'], ascending=['True','True']).reset_index(drop=True)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    data['Customers'] = data.groupby(['Store','DayOfWeek'])['Customers'].transform(lambda x: x.fillna(x.mean()))
    data['Sales'] = data.groupby(['Store','DayOfWeek'])['Sales'].transform(lambda x: x.fillna(x.mean()))

    old_cols = data.columns
    data = data.dropna(axis=1)
    new_cols = data.columns
    print('dropping {} columns due to nulls'.format(len(old_cols) - len(new_cols)))
    print('those cols are {}'.format(set(old_cols).difference(set(new_cols))))

    data = data.drop('DayOfWeek', axis=1)
    data = add_datetime_features_day_of_week(data)
    data = add_datetime_features_week(data)
    print(' ')
    print('data shape before split {}'.format(data.shape))
    print(data.loc[:, 'Date'].iloc[0], data.loc[:, 'Date'].iloc[-1])

    lag_column = 'Sales'
    lags = 2
    #  hack because lags need the date
    #  maybe best to do lags first - not a big deal
    data2 = lag_all_stores(data, lag_column, lags)
    #  need to rename to get the drop to work in split_dataset
    data2 = data2.rename({'Sales-lag-0': 'Sales'}, axis=1)
    data2 = data2.dropna(subset=['Sales-lag-1','Sales-lag-2'], axis=0).reset_index(drop=True)

    data = data.drop('Date', axis=1)
    # data2 = data.drop('Date', axis=1) - this is done in the lagging

    def split_dataset(data, base, validation_sets):
        data = data.copy()
        print('starting {}'.format(base))
        tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)
        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            fold_name = 'fold' + str(fold)
            os.makedirs(base + fold_name, exist_ok=True)

            data.iloc[train_index,:].drop('Sales',axis=1).to_csv(base + fold_name + '/' + '_train_X.csv', index=False)
            data.iloc[train_index,:]['Sales'].to_csv(base + fold_name + '/' + '_train_y.csv', index=False)
            data.iloc[test_index,:].drop('Sales',axis=1).to_csv(base + fold_name + '/' + '_test_X.csv', index=False)
            data.iloc[test_index,:]['Sales'].to_csv(base + fold_name + '/' + '_test_y.csv', index=False)
        print('done for {}'.format(base))

    split_dataset(data, 'data/scenario_1_control/', validation_sets)
    split_dataset(data2, 'data/scenario_2_lags', validation_sets)
    split_dataset(data, 'data/scenario_3_pred/', validation_sets)
