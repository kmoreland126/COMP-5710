import numpy as np
import pymannkendall as pmk

def understandTrends(data_, categ):
    try:
        trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
        print('For {} trend is {} and p-value is {}'.format(categ, trend, p))
    except Exception as e:
        print('Error processing {}: {}'.format(categ, e))

# Test Case 10: Data with non-numeric values
data10 = [1, 2, 'three', 4]
understandTrends(data10, 'Non-Numeric Data')
