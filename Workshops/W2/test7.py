import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 7: Large, alternating data with no overall trend
data7 = [np.sin(x) for x in np.arange(0, 50, 0.5)]
understandTrends(data7, 'Alternating Data')
