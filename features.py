import pandas as pd
import numpy as np

#  factor out yesterdays sales (target engineering)
#  holidays
#  linear trend

def lag_all_stores(raw, column, lags):
    out = []
    stores = raw.Store.unique()
    assert len(stores) == max(stores) + 1  # is this right???
    for store in stores:
        print('Processing Store ' + str(store))
        store_dataframe = raw[raw['Store'] == store]
        store_dataframe = add_lags_to_single_store(store_dataframe, column, lags)
        out.append(store_dataframe)

    return pd.concat(out, axis=0)

def add_lags_to_single_store(raw, column, lags):

    assert len(set(raw.loc[:,'Store'])) == 1
    raw = raw.sort_values(by='Date', ascending=True)

    data = raw.loc[:, column]
    out = [data.shift(l) for l in range(lags+1)]
    out = pd.concat(out, axis=1)
    out.columns = [column+'-lag-'+str(l) for l in range(lags+1)]

    remaining = raw.loc[:, raw.columns != column]
    out = pd.concat([remaining, out], axis=1)
    return out


def add_sales_per_customer(historical, test):
    """adds the historical sales, customers & sales per customer"""
    #  load historical - use this in data.py
    # historical = pd.read_csv('./data/raw/train.csv')

    data = historical.groupby('Store').mean()
    data.loc[:, 'sales-per-customer'] = data.loc[:, 'Sales'] / data.loc[:, 'Customers']
    data = data.loc[:, ['Sales', 'Customers', 'sales-per-customer']]
    data.columns = ['sales', 'customers', 'sales-per-customer']
    test = test.merge(data, on='Store')
    return test

def add_datetime_features_day_of_week(data):
    data['day'] = pd.to_datetime(data.loc[:, 'Date'])
    day = data['day'].dt.dayofweek
    data['day-of-week-sin'] = np.sin(2 * np.pi * day/23.0)
    data['day-of-week-cos'] = np.cos(2 * np.pi * day/23.0)
    return data.drop('day', axis=1)


def add_datetime_features_week(data):
    data['week'] = pd.to_datetime(data.loc[:, 'Date'])
    week = data['week'].dt.week
    data['month-sin'] = np.sin(2 * np.pi * week/51.0)
    data['month-cos'] = np.cos(2 * np.pi * week/51.0)
    return data.drop('week', axis=1)


