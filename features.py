import pandas as pd
import numpy as np

#  sales per customer


#  lagged values

def add_lags(raw, column, lags):

    data = raw.loc[:, 'A']
    out = [data.shift(l) for l in range(lags+1)]
    out = pd.concat(out, axis=1)
    out.columns = [column+'-lag-'+str(l) for l in range(lags+1)]

    remaining = raw.loc[:, raw.columns != column]
    out = pd.concat([remaining, out], axis=1)
    return out

test = pd.DataFrame(
    {'A': [10, 20, 30, 40, 50],
     'B': [100, 200, 300, 400, 500]}
)

def test_lagged_values():

    data = add_lags(test, 'A', lags=2)
    np.testing.assert_array_equal(data.loc[:, 'A-lag-1'].values, np.array([np.nan, 10, 20, 30, 40]))
    np.testing.assert_array_equal(data.loc[:, 'A-lag-2'].values, np.array([np.nan, np.nan, 10, 20, 30]))
    np.testing.assert_array_equal(data.loc[:, 'B'].values, test.loc[:, 'B'].values)


out = add_lags(test, 'A', 2)
