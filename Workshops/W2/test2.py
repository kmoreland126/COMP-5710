import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 2: Data with a clear increasing trend
data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
understandTrends(data2, 'Increasing Trend')
