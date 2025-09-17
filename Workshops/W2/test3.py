import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 3: Data with a clear decreasing trend
data3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
understandTrends(data3, 'Decreasing Trend')
