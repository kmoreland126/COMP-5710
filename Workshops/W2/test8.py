import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 8: Data with a single outlier
data8 = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]
understandTrends(data8, 'Data with Outlier')
