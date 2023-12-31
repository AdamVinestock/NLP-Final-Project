"""
Script to read log-perplexity data of many sentences and characterize the empirical distribution.
We also report the mean log-perplexity as a function of sentence length
"""

import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
import logging
import numpy as np


def fit_survival_func(xx, log_space=True):
    """
    Returns an estimated survival function to the data in :xx: using
    interpolation.

    Args:
        :xx:  data
        :log_space:  indicates whether fitting is in log space or not.

    Returns:
         univariate function
    """
    assert len(xx) > 0

    eps = 1 / len(xx)
    inf = 1 / eps

    sxx = np.sort(xx)
    qq = np.mean(np.expand_dims(sxx,1) >= sxx, 0)

    if log_space:
        qq = -np.log(qq)


    if log_space:
        return interp1d(sxx, qq, fill_value=(0 , np.log(inf)), bounds_error=False)
    else:
        return interp1d(sxx, qq, fill_value=(1 , 0), bounds_error=False)


def fit_per_length_survival_function(lengths, xx, G=501, log_space=True):
    """
    Use 2D interpolation over the empirical survival function of the pairs (length, x)

    Args:
        :lengths:, :xx:, 1-D arrays
        :G:  number of grid points to use in the interpolation in the xx dimension
        :log_space:  indicates whether result is in log space or not.

    Returns:
        bivariate function (length, x) -> [0,1]
    """

    assert len(lengths) == len(xx)

    assert not np.isnan(lengths).any() and not np.isnan(xx).any()   # To delete!!!!!!!!!!!!!
    assert not np.isinf(lengths).any() and not np.isinf(xx).any()

    min_tokens_per_sentence = lengths.min()
    max_tokens_per_sentence = lengths.max()
    ll = np.arange(min_tokens_per_sentence, max_tokens_per_sentence)

    ppx_min_val = xx.min()
    ppx_max_val = xx.max()
    xx0 = np.linspace(ppx_min_val, ppx_max_val, G)

    ll_valid = []
    zz = []
    for i, l in enumerate(ll):
        xx1 = xx[lengths == l]
        if len(xx1) > 0:
            univariate_survival_func = fit_survival_func(xx1, log_space=log_space)
            assert not np.isnan(univariate_survival_func(xx0)).any()  # To delete!!!!!!!!!!!!!
            ll_valid.append(l)
            zz.append(univariate_survival_func(xx0))

    test_zz = np.vstack(zz)
    assert not np.isnan(test_zz).any()
    assert not np.isinf(test_zz).any()

    func = RectBivariateSpline(np.array(ll_valid), xx0, np.vstack(zz))

    test_val = func(ll_valid[0], xx0[0])
    print(test_val)

    if log_space:
        def func2d(x, y):
            return np.exp(-func(x,y))
        return func2d
    else:
        return func