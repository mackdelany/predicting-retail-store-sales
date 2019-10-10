import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

from features import lag_all_stores, add_lags_to_single_store, add_sales_per_customer, add_datetime_features_day_of_week, add_datetime_features_week, mean_encode


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

    for i, store_number in enumerate(train.Store):
        if pd.isna(store_number) == True:
            if np.absolute(train.loc[i-1, 'Store'] - train.loc[i+1, 'Store']) == 2:
                train.loc[i, 'Store'] = (train.loc[i-1, 'Store'] + train.loc[i+1, 'Store'] / 2)

    data = train[~train['Store'].isna()]
    assert sum(data.loc[:, 'Store'].isnull()) == 0
    print('train shape {}'.format(data.shape))

    data = add_sales_per_customer(data, data)

    #  why merge with stores here?
    data = data.merge(store, how='left', left_on='Store', right_on='Store')
    data = data.sort_values(by=['Date','Store'], ascending=['True','True']).reset_index(drop=True)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    data['Customers'] = data.groupby(['Store','DayOfWeek'])['Customers'].transform(lambda x: x.fillna(x.mean()))
    data['Sales'] = data.groupby(['Store','DayOfWeek'])['Sales'].transform(lambda x: x.fillna(x.mean()))

    assert sum(data.loc[:, 'Store'].isnull()) == 0
    data = mean_encode(data, col='Store', on='Sales')
    assert sum(data.loc[:, 'Store'].isnull()) == 0

    cleanup = {
        "StateHoliday": {'0': 0, 'a': 1, 'b': 2, 'c': 3},
        "Assortment": {'a': 0, 'b': 1, 'c': 2},
        "StoreType": {'a': 0, 'b': 1, 'c': 2, 'd': 3},
        "PromoInterval": {'no-promo': 0, 'Feb,May,Aug,Nov': 1, 'Jan,Apr,Jul,Oct': 2, 'Mar,Jun,Sept,Dec': 3}
    }
    data.replace(cleanup, inplace=True)

    data = data.drop('DayOfWeek', axis=1)
    # data = add_datetime_features_day_of_week(data)
    # data = add_datetime_features_week(data)
    data.loc[:, 'month'] = data.loc[:, 'Date'].dt.month
    data.loc[:, 'week'] = data.loc[:, 'Date'].dt.week
    data.loc[:, 'day-of-week'] = data.loc[:, 'Date'].dt.dayofweek
    assert sum(data.loc[:, 'Store'].isnull()) == 0

    #  drop zero target
    print('dropping target')
    print('train shape before drop of zero sales {}'.format(data.shape))
    mask = data.loc[:, 'Sales'] != 0
    data = data.loc[mask, :]
    data = data.dropna(subset=['Sales'], axis=0)
    print('train shape after drop of zero sales {}'.format(data.shape))

    assert sum(data.loc[:, 'Store'].isnull()) == 0
    for col in data.columns:
        print(col, ' - ', sum(data.loc[:, col].isnull()))

    fill_with_token = ['Promo', 'StateHoliday']

    for tok in fill_with_token:
        data.loc[:, tok] = data.loc[:, tok].fillna(0)
        assert sum(data.loc[:, tok].isnull()) == 0

    import pdb; pdb.set_trace()
    old_cols = data.columns
    data = data.dropna(axis=1)
    new_cols = data.columns
    print('dropping {} columns due to nulls'.format(len(old_cols) - len(new_cols)))
    print('those cols are {}'.format(set(old_cols).difference(set(new_cols))))
    print('train shape after drop {}'.format(data.shape))

    print(' ')
    print('data shape before split {}'.format(data.shape))
    print(data.loc[:, 'Date'].iloc[0], data.loc[:, 'Date'].iloc[-1])

    assert(sum(data.loc[:, 'Sales'] == 0)) == 0

    lag_column = 'Sales'
    lags = 2
    #  hack because lags need the date
    #  maybe best to do lags first - not a big deal
    data2 = lag_all_stores(data, lag_column, lags)
    #  need to rename to get the drop to work in split_dataset
    data2 = data2.rename({'Sales-lag-0': 'Sales'}, axis=1)
    data2 = data2.dropna(subset=['Sales-lag-1','Sales-lag-2'], axis=0).reset_index(drop=True)

    data = data.drop('Date', axis=1)
    data2 = data2.drop('Date', axis=1)

    def split_dataset(data, base, validation_sets):
        data = data.copy()
        print('starting {}'.format(base))
        # tscv = TimeSeriesSplit(max_train_size=round(max_train_size*data.shape[0]), n_splits=validation_sets)
        tscv = TimeSeriesSplit(n_splits=validation_sets)
        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            fold_name = 'fold' + str(fold)
            os.makedirs(base + fold_name, exist_ok=True)

            data.iloc[train_index,:].drop('Sales',axis=1).to_csv(base + fold_name + '/' + 'train_X.csv', index=False)
            data.iloc[train_index,:]['Sales'].to_csv(base + fold_name + '/' + 'train_y.csv', index=False, header=True)
            data.iloc[test_index,:].drop('Sales',axis=1).to_csv(base + fold_name + '/' + 'test_X.csv', index=False)
            data.iloc[test_index,:]['Sales'].to_csv(base + fold_name + '/' + 'test_y.csv', index=False, header=True)
        print('done for {}'.format(base))

    split_dataset(data, 'data/scenario_1_control/', validation_sets)
    split_dataset(data2, 'data/scenario_2_lags/', validation_sets)
    split_dataset(data, 'data/scenario_3_pred/', validation_sets)
