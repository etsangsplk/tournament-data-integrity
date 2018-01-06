import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from integrity.util import upper_triangle, logloss_by_era
from integrity.log import interval, array_interval, _assert


def check(data):
    err_count = 0
    err_count += header(data)
    err_count += ids(data)
    err_count += eras(data)
    err_count += regions(data)
    err_count += features(data)
    err_count += labels(data)
    err_count += predictions(data)
    return err_count


def header(data):

    logging.info('HEADER')

    err_count = 0

    # train and tournment csv files should have the same header
    if (data.header['train'] != data.header['tournament']).any():
        err_count += 1
        logging.warn('train and tournament csv files have different headers')

    # columns should be in the correct order. We are especially concerned with
    # the order of the features which should be feature1, feature2, ... and
    # not feature1, feature10, feature11, ...
    header = ['id', 'era', 'data_type']
    header += ['feature'+str(i) for i in range(1, 51)]
    header += ['target']
    for i in range(len(header)):
        err_count += _assert('header column',
                             data.header['train'][i], '==', header[i])

    # should have the correct number of columns
    actual = len(data.header['train'])
    desired = len(header)
    err_count += _assert('number of column in csv file', actual, '==', desired)

    return err_count


def ids(data):

    logging.info('IDS')

    err_count = 0

    # duplicate ids
    num_duplicate = data.ID.size - np.unique(data.ID).size
    err_count += _assert('duplicate ids', num_duplicate, '==', 0)

    return err_count


def eras(data):

    logging.info('ERAS')

    err_count = 0

    # number of eras
    target = {'train': 120,
              'validation': 12,
              'test': 1,
              'live': 1}
    for region in target:
        n = np.unique(data.era[data.region == region]).size
        msg = 'number of eras in %s' % region
        err_count += _assert(msg, n, '==', target[region])

    return err_count


def regions(data):

    logging.info('REGIONS')

    err_count = 0

    # make sure all regions are present and there are no extra regions
    target = set(['train', 'validation', 'test', 'live'])
    regions = set(np.unique(data.region))
    if regions != target:
        diff = regions - target
        if len(diff) > 0:
            err_count += 1
            logging.warn('extra regions found: %s' % str(diff))
        diff = target - regions
        if len(diff) > 0:
            err_count += 1
            logging.warn('missing regions: %s' % str(diff))

    return err_count


def features(data):

    logging.info('FEATURES')

    err_count = 0

    # nonfinite feature values
    num = (~np.isfinite(data.x)).sum()
    err_count += _assert('nonfinite feature values', num, '==', 0)

    # abs correlation of features
    corr = np.corrcoef(data.x.T)
    corr = upper_triangle(corr)
    corr = np.abs(corr)
    err_count += interval('mean abs corr of features',
                          corr.mean(), [0.18, 0.22])
    err_count += interval('max  abs corr of features',
                          corr.max(), [0.7, 0.73])

    # distribution of each feature in each era
    for era, feature_num, x in data.era_feature_iter():

        msg = 'range of feature %2d in %s' % (feature_num, era.ljust(6))
        err_count += array_interval(msg, x, [0, 1])

        msg = 'mean  of feature %2d in %s' % (feature_num, era.ljust(6))
        err_count += interval(msg, x.mean(), [0.44, 0.57])

        msg = 'std   of feature %2d in %s' % (feature_num, era.ljust(6))
        err_count += interval(msg, x.std(), [0.085, 0.158])

        msg = 'skewn of feature %2d in %s' % (feature_num, era.ljust(6))
        skew = ((x - x.mean())**3).mean() / x.std()**3
        err_count += interval(msg, skew, [-0.44, 0.44])

        msg = 'kurto of feature %2d in %s' % (feature_num, era.ljust(6))
        kurt = ((x - x.mean())**4).mean() / x.std()**4
        err_count += interval(msg, kurt, [2.44, 3.7])

    return err_count


def labels(data):

    logging.info('LABELS')

    err_count = 0

    # labels should only contain 0 and 1
    idx = data.nonmissing_label_index()
    y = data.y[idx]
    idx = np.logical_or(y == 0, y == 1)
    err_count += _assert("number of non 0, 1 labels",
                         idx.size - idx.sum(), '==', 0)

    # test and live labels should be NaN
    regions = ['test', 'live']
    for region in regions:
        idx = data.region == region
        y = data.y[idx]
        if not np.isnan(y).all():
            err_count += 1
            logging.warn("Some %s labels are not NaN" % region)

    # mean of labels and number of labels
    y_mean = []
    for era, index in data.era_iter():

        y = data.y[index]

        # labels are missing in eraX
        if era != 'eraX':
            msg = 'mean of labels in %s' % era.ljust(6)
            ym = y.mean()
            err_count += interval(msg, ym, [0.499, 0.501])
            y_mean.append(ym)

        msg = 'num of labels in %s' % era.ljust(6)
        if era == 'eraX':
            limit = [196000, 198000]
        else:
            limit = [1500, 4200]
        err_count += interval(msg, y.size, limit)

    # label bias
    msg = 'fraction of eras with label mean less than half'
    y_mean = np.array(y_mean)
    err_count += interval(msg, (y_mean < 0.5).mean(), [0.4, 0.6])

    return err_count


def predictions(data):

    logging.info('PREDICTIONS')

    err_count = 0

    # fit logistic regression model on train data
    idx = data.region == 'train'
    xtrain = data.x[idx]
    ytrain = data.y[idx]
    eratrain = data.era[idx]
    clf = LogisticRegression()
    clf.fit(xtrain, ytrain)

    # predict using train data
    yhat_train = clf.predict_proba(xtrain)[:, 1]

    # check train logloss and consistency
    logloss = log_loss(ytrain, yhat_train)
    err_count += interval('train logloss', logloss, [0.691, 0.693])
    loglosses = logloss_by_era(eratrain, ytrain, yhat_train)
    consistency = (loglosses < np.log(2)).mean()
    err_count += interval('train consistency', consistency, [0.57, 0.84])

    # predict using validation data
    yvalid, yhat = calc_yhat('validation', clf, data)

    # check validation logloss and consistency
    logloss = log_loss(yvalid, yhat)
    err_count += interval('validation logloss', logloss, [0.691, 0.693])
    idx = data.region == 'validation'
    loglosses = logloss_by_era(data.era[idx], yvalid, yhat)
    consistency = (loglosses < np.log(2)).mean()
    err_count += interval('validation consistency', consistency, [0.5, 0.84])

    # check test and live predictions
    for region in ('test', 'live'):
        y, yhat = calc_yhat(region, clf, data)
        target = [0.99 * yhat_train.min(), 1.01 * yhat_train.max()]
        msg = 'predictions in %s region'
        err_count += array_interval(msg % region, yhat, target)

    return err_count


def calc_yhat(region, clf, data):
    idx = data.region == region
    x = data.x[idx]
    y = data.y[idx]
    yhat = clf.predict_proba(x)[:, 1]
    return y, yhat
