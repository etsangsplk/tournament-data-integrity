import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ni.util import upper_triangle


def check(data):
    ids(data)
    eras(data)
    regions(data)
    features(data)
    labels(data)
    predictions(data)


def ids(data):

    logging.info('IDS')

    # duplicate ids
    num_duplicate = data.ID.size - np.unique(data.ID).size
    _assert('duplicate ids', num_duplicate, '==', 0)


def eras(data):

    logging.info('ERAS')

    # number of eras
    target = {'train': 85,
              'validation': 12,
              'test': 1,
              'live': 1}
    for region in target:
        n = np.unique(data.era[data.region == region]).size
        msg = 'number of eras in %s' % region
        _assert(msg, n, '==', target[region])


def features(data):

    logging.info('FEATURES')

    # nonfinite feature values
    n = (~np.isfinite(data.x)).sum()
    _assert('nonfinite feature values', n, '==', 0)

    # abs correlation of features
    corr = np.corrcoef(data.x.T)
    corr = upper_triangle(corr)
    corr = np.abs(corr)
    interval('mean abs corr of features', corr.mean(), [0.1, 0.2])
    interval('max  abs corr of features', corr.max(), [0.6, 0.7])

    # distribution of each feature in each era
    for era, feature_num, x in data.era_feature_iter():

        msg = 'range of feature %2d in %s' % (feature_num, era.ljust(6))
        array_interval(msg, x, [0, 1])

        msg = 'mean  of feature %2d in %s' % (feature_num, era.ljust(6))
        interval(msg, x.mean(), [0.4545, 0.5505])

        msg = 'std   of feature %2d in %s' % (feature_num, era.ljust(6))
        interval(msg, x.std(), [0.09, 0.14])

        msg = 'skewn of feature %2d in %s' % (feature_num, era.ljust(6))
        skew = ((x - x.mean())**3).mean() / x.std()**3
        interval(msg, skew, [-0.4, 0.4])

        msg = 'kurto of feature %2d in %s' % (feature_num, era.ljust(6))
        kurt = ((x - x.mean())**4).mean() / x.std()**4
        interval(msg, kurt, [2.5, 3.5])


def regions(data):

    logging.info('REGIONS')

    # make sure all regions are present and there are no extra regions
    target = set(['train', 'validation', 'test', 'live'])
    regions = set(np.unique(data.region))
    if regions != target:
        diff = regions - target
        if len(diff) > 0:
            logging.warn('extra regions found: %s' % str(diff))
        diff = target - regions
        if len(diff) > 0:
            logging.warn('missing regions: %s' % str(diff))


def labels(data):

    logging.info('LABELS')

    # labels should only contain 0 and 1
    idx = data.nonmissing_label_index()
    y = data.y[idx]
    idx = np.logical_or(y == 0, y == 1)
    _assert("number of non 0, 1 labels", idx.size - idx.sum(), '==', 0)

    # mean of labels and number of labels
    y_mean = []
    for era, index in data.era_iter():

        y = data.y[index]

        msg = 'mean of labels in %s' % era.ljust(6)
        ym = y.mean()
        interval(msg, ym, [0.499, 0.501])
        y_mean.append(ym)

        msg = 'num  of labels in %s' % era.ljust(6)
        if era == 'eraX':
            limit = [270000, 280000]
        else:
            limit = [5940, 6750]
        interval(msg, y.size, limit)

    # label bias
    msg = 'fraction of eras with label mean less than half'
    y_mean = np.array(y_mean)
    interval(msg, (y_mean < 0.5).mean(), [0.4, 0.6])


def predictions(data):

    logging.info('PREDICTIONS')

    # fit logistic regression model on train data
    idx = data.region == 'train'
    xtrain = data.x[idx]
    ytrain = data.y[idx]
    eratrain = data.era[idx]
    clf = LogisticRegression()
    clf.fit(xtrain, ytrain)

    # predict using train data
    yhat = clf.predict_proba(xtrain)[:, 1]

    # check train logloss and consistency
    logloss = log_loss(ytrain, yhat)
    interval('train logloss', logloss, [0.68, 0.688])
    loglosses = logloss_by_era(eratrain, ytrain, yhat)
    consistency = (loglosses < np.log(2)).mean()
    interval('train consistency', consistency, [0.57, 0.84])

    # predict using validation data
    idx = data.region == 'validation'
    xvalid = data.x[idx]
    yvalid = data.y[idx]
    eravalid = data.era[idx]
    yhat = clf.predict_proba(xvalid)[:, 1]

    # check validation logloss and consistency
    logloss = log_loss(yvalid, yhat)
    interval('validation logloss', logloss, [0.68, 0.688])
    loglosses = logloss_by_era(eravalid, yvalid, yhat)
    consistency = (loglosses < np.log(2)).mean()
    interval('validation consistency', consistency, [0.57, 0.84])


def logloss_by_era(era, y, yhat):
    unique_eras = np.unique(era)
    n = unique_eras.size
    logloss = np.zeros(n)
    for i in range(n):
        idx = era == unique_eras[i]
        yi = y[idx]
        yh = yhat[idx]
        logloss[i] = log_loss(yi, yh)
    return logloss


# ---------------------------------------------------------------------------
# logging utilities

TAB = '  '


def interval(message, value, limit, level='warn'):
    if value < limit[0] or value > limit[1]:
        log = get_logger(level)
        fmt = TAB + message + " %7.4f not in %s"
        log(fmt % (value, str(limit)))


def array_interval(message, arr, limit, level='warn'):
    amin, amax = arr.min(), arr.max()
    if amin < limit[0] or amax > limit[1]:
        log = get_logger(level)
        fmt = TAB + message + " [%7.4f, %7.4f] not in %s"
        log(fmt % (amin, amax, str(limit)))


def _assert(message, value, op, target, level='warn'):
    oppo = {'==': '!=',
            '!=': '==',
            '>': '<=',
            '<=': '>',
            '<': '>=',
            '>=': '<'}
    if op not in oppo:
        raise ValueError('operation `op` is not recognized')
    if op == '==':
        failed = value != target
    elif op == '!=':
        failed = value == target
    elif op == '>':
        failed = value <= target
    elif op == '<=':
        failed = value > target
    elif op == '<':
        failed = value >= target
    elif op == '<=':
        failed = value < target
    if failed:
        log = get_logger(level)
        fmt = TAB + message + " %7.4f %s %s"
        log(fmt % (value, oppo[op], str(target)))


def get_logger(level):
    if level == 'info':
        log = logging.info
    elif level == 'warn':
        log = logging.warn
    elif level == 'error':
        log = logging.error
    elif level == 'critical':
        log = logging.critical
    else:
        raise ValueError("logging `level` not recognized")
    return log
