from features import *


def test_lagged_values():
    test = pd.DataFrame(
        {'A': [10, 20, 30, 40, 50],
         'B': [100, 200, 300, 400, 500],
         'Store': [1] * 5,
         'Date': pd.date_range(start='2018-01-01', periods=5, freq='1d')}
    )
    data = add_lags_to_single_store(test, 'A', lags=2)
    np.testing.assert_array_equal(data.loc[:, 'A-lag-1'].values, np.array([np.nan, 10, 20, 30, 40]))
    np.testing.assert_array_equal(data.loc[:, 'A-lag-2'].values, np.array([np.nan, np.nan, 10, 20, 30]))
    np.testing.assert_array_equal(data.loc[:, 'B'].values, test.loc[:, 'B'].values)


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


def test_lag_all_stores():
    test1 = pd.DataFrame(
        {'A': [10, 20, 30, 40, 50],
         'B': [100, 200, 300, 400, 500],
         'Store': [0] * 5,
         'Date': pd.date_range(start='2018-01-01', periods=5, freq='1d')}
    )

    test2 = pd.DataFrame(
        {'A': [10, 20, 30, 40, 50],
         'B': [100, 200, 300, 400, 500],
         'Store': [1] * 5,
         'Date': pd.date_range(start='2018-01-01', periods=5, freq='1d')}
    )

    test = pd.concat([test1, test2], axis=0)

    out = lag_all_stores(test, 'B', 2)

    one_test = out[out.loc[:, 'Store'] == 0]
    np.testing.assert_array_equal(one_test.loc[:, 'B-lag-1'].values, np.array([np.nan, 100, 200, 300, 400]))
    np.testing.assert_array_equal(one_test.loc[:, 'B-lag-2'].values, np.array([np.nan, np.nan, 100, 200, 300]))


    out = lag_all_stores(test, 'A', 2)
    two_test = out[out.loc[:, 'Store'] == 1]
    np.testing.assert_array_equal(two_test.loc[:, 'A-lag-1'].values, np.array([np.nan, 10, 20, 30, 40]))
    np.testing.assert_array_equal(two_test.loc[:, 'A-lag-2'].values, np.array([np.nan, np.nan, 10, 20, 30]))


def test_mean_encoding():

    store1 = pd.DataFrame(
        {'store': ['A'] * 3,
         'Sales': [100, 200, 300],
         'noise': [0, 0, 0]}
    )

    store2 = pd.DataFrame(
        {'store': ['B'] * 3,
         'Sales': [10, 20, 30],
         'noise': [0, 0, 0]}
    )

    data = pd.concat([store1, store2], axis=0)

    data = mean_encode(data, col='store', on='Sales')

    np.testing.assert_array_equal(data.loc[:, 'store'], np.array([200, 200, 200, 20, 20, 20]))
