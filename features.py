import pandas as pd
import numpy as np

#  factor out yesterdays sales (target engineering)
#  holidays



def lag_all_stores(raw, column, lags):

    raw['Sales-lag-1'] = np.nan
    raw['Sales-lag-2'] = np.nan

    for store in raw.Store.unique():

        print('Processing Store ' + str(store))

        store_dataframe = raw[raw['Store'] == store]
        store_dataframe = add_lags_to_single_store(store_dataframe.drop(['Sales-lag-1','Sales-lag-2'], axis=1), column, 2)

        raw.loc[raw['Store'] == store, 'Sales-lag-1'] = store_dataframe['Sales-lag-1']
        raw.loc[raw['Store'] == store, 'Sales-lag-2'] = store_dataframe['Sales-lag-2']

    return raw



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

def test_lagged_values():
    test = pd.DataFrame(
        {'A': [10, 20, 30, 40, 50],
         'B': [100, 200, 300, 400, 500]}
    )
    data = add_lags(test, 'A', lags=2)
    np.testing.assert_array_equal(data.loc[:, 'A-lag-1'].values, np.array([np.nan, 10, 20, 30, 40]))
    np.testing.assert_array_equal(data.loc[:, 'A-lag-2'].values, np.array([np.nan, np.nan, 10, 20, 30]))
    np.testing.assert_array_equal(data.loc[:, 'B'].values, test.loc[:, 'B'].values)


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

def test_sales_per_customer():
    historical = pd.DataFrame(
        {'Store': ['A', 'B', 'B', 'A'],
         'Sales': [200, 300, 100, 50],
         'Customers': [20, 30, 10, 50]}
    )

    test = pd.DataFrame(
        {'Store': ['B', 'A']}
    )

    out = add_sales_per_customer(historical, test)
    np.testing.assert_array_equal(out.loc[:, 'sales'].values, np.array([200, 125]))
    np.testing.assert_array_equal(out.loc[:, 'customers'].values, np.array([20, 35]))
    np.testing.assert_array_equal(out.loc[:, 'sales-per-customer'].values, np.array([(300+100)/(30+10), (200+50)/(20+50)]))


def add_datetime_features_day_of_week(data):
    data.index = pd.to_datetime(data.loc[:, 'Date'])
    day = data.index.dayofweek
    data['day-of-week-sin'] = np.sin(2 * np.pi * day/23.0)
    data['day-of-week-cos'] = np.cos(2 * np.pi * day/23.0)
    return data


def add_datetime_features_week(data):
    data.index = pd.to_datetime(data.loc[:, 'Date'])
    week = data.index.week
    # import pdb; pdb.set_trace()
    data['month-sin'] = np.sin(2 * np.pi * week/52.0)
    data['month-cos'] = np.cos(2 * np.pi * week/52.0)
    return data

#  linear trend

def test_day_of_week():
    test_index = pd.DatetimeIndex(start='01/01/2018', end='31/12/2018', freq='d')
    test_df = pd.DataFrame(np.random.uniform(size=test_index.shape[0]), index=test_index)
    test_df.loc[:, 'Date'] = test_index

    features = add_datetime_features_day_of_week(test_df)

    features = features.loc[:, ['day-of-week-sin', 'day-of-week-cos']]

    assert features.iloc[0, :].values.all() == features.iloc[7, :].values.all()
    assert features.iloc[14, :].values.all() == features.iloc[21, :].values.all()
    assert not np.array_equal(features.iloc[:, 0].values, features.iloc[:, 1].values)

    #  check that each value is unique for the entire day
    assert not features.iloc[:6, :].duplicated().any()
    #  check that we do have unique values across a longer time period
    assert features.iloc[:, :].duplicated().any()


def test_month():
    test_index = pd.DatetimeIndex(start='01/01/2018', end='31/12/2018', freq='d')
    test_df = pd.DataFrame(np.random.uniform(size=test_index.shape[0]), index=test_index)
    test_df.loc[:, 'Date'] = test_index

    features = add_datetime_features_week(test_df)

    features = features.loc[:, ['month-sin', 'month-cos']]

    assert features.iloc[0, :].values.all() == features.iloc[32, :].values.all()
    assert features.iloc[42, :].values.all() == features.iloc[72, :].values.all()
    assert not np.array_equal(features.iloc[:, 0].values, features.iloc[:, 1].values)

    #  check that we do have unique values across a longer time period
    assert features.iloc[:, :].duplicated().any()
