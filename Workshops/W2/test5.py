import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 5: Short data series
data5 = [1, 5, 2]
understandTrends(data5, 'Short Data Series')
