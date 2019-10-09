import pandas as pd
import numpy as np

#  factor out yesterdays sales (target engineering)
#  holidays

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
    data['day-of-week-sin'] = np.sin(2 * np.pi * dataset[PresentDateTime]/23.0)
    data['day-of-week-cos'] = np.cos(2 * np.pi * dataset[PresentDateTime]/23.0)
    return data


def add_datetime_features_day_of_week(data):
    data.index = pd.to_datetime(data.loc[:, 'Date'])
    data['day-of-week-sin'] = np.sin(2 * np.pi * dataset[PresentDateTime]/23.0)
    data['day-of-week-cos'] = np.cos(2 * np.pi * dataset[PresentDateTime]/23.0)
    return data

#  day of week, month of year, linear trend


