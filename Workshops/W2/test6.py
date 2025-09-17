import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 6: Small, random data with no trend
data6 = [3, 1, 4, 1, 5, 9]
understandTrends(data6, 'Small Data, No Trend')
