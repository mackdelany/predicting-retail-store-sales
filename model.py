from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xgboost as xgb

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def plot_feature_importances(model, num_features, model_dir):
    data = model.get_score(importance_type='gain')
    data = pd.DataFrame(data, index=['importances'])
    data.loc['features', :] = data.columns
    data = data.transpose()
    data.sort_values('importances', inplace=True)

    f, a = plt.subplots()
    data.plot(ax=a, kind='bar', x='features', y='importances')
    plt.gcf().subplots_adjust(bottom=0.3)
    f.savefig(os.path.join(model_dir, 'importances.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='local', nargs='?')
    parser.add_argument('--cpu', default=8, nargs='?')
    args = parser.parse_args()

    scenario = 'scenario_2_lags'
    fold = 2
    dataset = [
        'train_X', 'train_Y', 'test_X', 'test_Y'
    ]
    path = Path('./data') / scenario / 'fold{}'.format(fold) 
    dataset = {
        d: pd.read_csv(path / '{}.csv'.format(d))
        for d in dataset
    }
    print(fold)
    from collections import defaultdict
    res = defaultdict(list)

    x_tr = dataset['train_X']
    x_te = dataset['test_X']
    y_tr = dataset['train_Y']
    y_te = dataset['test_Y']

    print('train')
    print(x_tr.shape, y_tr.shape)
    print('test')
    print(x_te.shape, y_te.shape)

    print(x_tr.columns)

    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dtest = xgb.DMatrix(x_te, label=y_te)

    params = {
        'verbosity': 1,
        'nthread': 5,
        'eta': 0.2, # learning rate, 0.3
        'max_depth': 3, # 6
        'num_round': 500,
    }
    num_round = params.pop('num_round')

    m = xgb.train(
            params,
            dtrain,
            num_round,
            [(dtest, 'eval'), (dtrain, 'train')], verbose_eval=50
    )

    plot_feature_importances(m, x_tr.shape[1], path)

    m.dump_model(str(path / 'model.raw.txt'))

    res['train-metric'].append(metric(
        m.predict(xgb.DMatrix(x_tr)),
        y_tr.values,
    ))
    res['test-metric'].append(metric(
        m.predict(xgb.DMatrix(x_te)),
        y_te.values,
    ))
    print(res)
    model_dir = Path('.')
    pd.DataFrame(res).to_csv(path / 'results.csv')
