from time import strptime
import pandas as pd
import numpy as np

#  factor out yesterdays sales (target engineering)
#  holidays
#  linear trend


def mean_encode(data, col, on):
    group = data.groupby(col).mean()
    mapper = {k: v for k, v in zip(group.index, group.loc[:, on].values)}
    data.loc[:, col] = data.loc[:, col].replace(mapper)
    data.loc[:, col].fillna(value=np.mean(data.loc[:, col]), inplace=True)
    return data


def lag_all_stores(raw, column, lags):
    out = []
    # stores = raw.Store.unique()
    stores = set(raw.loc[:, 'Store'])
    print('lagging stores')
    for store in stores:
        # print('Processing Store ' + str(store))
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
    data = data.loc[:, ['Customers', 'sales-per-customer']]
    data.columns = ['mean-customers', 'sales-per-customer']
    data.fillna({
        'mean-customers': np.mean(data.loc[:, 'mean-customers']),
        'sales-per-customer': np.mean(data.loc[:, 'sales-per-customer'])
    }, inplace=True)
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



def identify_competition_and_promo_start_date(store):

    idx_of_stores_with_no_competition = store[store['CompetitionDistance'].isnull()].index
    idx_of_stores_with_competition_always = store[store['CompetitionOpenSinceMonth'].isnull()][store.CompetitionDistance > 0].index

    mask = store.index.isin(idx_of_stores_with_no_competition.append(idx_of_stores_with_competition_always))

    idx_of_stores_where_competition_opened = store[~mask].index


    store.loc[idx_of_stores_with_no_competition,'competitionOpenDate'] = '01/01/2050'
    store.loc[idx_of_stores_with_competition_always,'competitionOpenDate'] = '01/01/1970'

    store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0).astype(int)

    for index in idx_of_stores_where_competition_opened:
        
        store.at[index,'competitionOpenDate'] = \
            (str(store.at[index,'CompetitionOpenSinceMonth']) + '/15/' + str(store.at[index,'CompetitionOpenSinceYear']))

    store['competitionOpenDate'] = pd.to_datetime(store['competitionOpenDate'])


    idx_of_stores_with_no_promo = store[store['Promo2SinceWeek'].isnull()].index

    mask = store.index.isin(idx_of_stores_with_no_promo)
    idx_of_stores_with_promos = store[~mask].index


    store.loc[idx_of_stores_with_no_promo,'promo2StartDate'] = '01/01/2050'

    store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0).astype(int)
    store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0).astype(int)

    for index in idx_of_stores_with_promos:
        store.loc[index,'promo2StartDate'] = (str(min(12,(((store.at[index,'Promo2SinceWeek'] * 7) // 30)+1)))\
                                                + '/15/' + str(store.at[index,'Promo2SinceYear']))
        
    store['promo2StartDate'] = pd.to_datetime(store['promo2StartDate'])

    return store





    

def identify_whether_promo2_running(merged_dataset):

    merged_dataset['PromoInterval'] = merged_dataset['PromoInterval'].fillna('no_promo').astype(str)

    def get_month_integers_from_month_strings(month_strings):

        if month_strings == 'no_promo':
            return 0
        else:
            month_array = []
            month_list = month_strings.split(",") 
            for month in month_list:
                if len(month) == 4:
                    month_array.append(9)
                else:
                    month_array.append(strptime(month,'%b').tm_mon)

            return month_array

    merged_dataset['promoMonths'] = merged_dataset['PromoInterval'].apply(get_month_integers_from_month_strings)

    def get_month(date):
        return date.month   

    merged_dataset['month_test'] = merged_dataset['Date'].apply(get_month)

    for index in range(merged_dataset.shape[0]):
        if isinstance(merged_dataset.at[index, 'promoMonths'],list) :
            if (merged_dataset.at[index, 'promo2StartDate'] <= merged_dataset.at[index, 'Date'])\
                    & (merged_dataset.at[index, 'month_test'] in merged_dataset.at[index, 'promoMonths'] ):
                merged_dataset.at[index, 'promoMonths'] = 1
        else :
            merged_dataset.at[index,'Promo2'] = 0
    
    merged_dataset['Promo2'] = merged_dataset['Promo2'].astype(int)

    return merged_dataset
