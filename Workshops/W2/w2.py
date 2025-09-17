import numpy as np
import pymannkendall as pmk

def understandTrends(data_):
	trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
	print('For {} trend is {} and p-value is {}'.format(categ, trend, p))

# Test Case 1: Random data with no trend
data1 = np.random.rand(50)
understandTrends(data1, 'Random Data')

#Test Case 2: Data with a clear increasing trend
data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
understandTrends(data2, 'Increasing Trend')

#Test Case 3: Data with a clear decreasing trend
data3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
understandTrends(data3, 'Decreasing Trend')

# Test Case 4: Data with repeated values (ties)
data4 = [1, 2, 2, 3, 4, 4, 4, 5]
understandTrends(data4, 'Data with Ties')

# Test Case 5: Short data series
data5 = [1, 5, 2]
understandTrends(data5, 'Short Data Series')

# Test Case 6: Small, random data with no trend
data6 = [3, 1, 4, 1, 5, 9]
understandTrends(data6, 'Small Data, No Trend')

# Test Case 7: Large, alternating data with no overall trend
data7 = [np.sin(x) for x in np.arange(0, 50, 0.5)]
understandTrends(data7, 'Alternating Data')

# Test Case 8: Data with a single outlier
data8 = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]
understandTrends(data8, 'Data with Outlier')

# Test Case 9: Empty data list
data9 = []
understandTrends(data9, 'Empty List')

# Test Case 10: Data with non-numeric values
data10 = [1, 2, 'three', 4]
understandTrends(data10, 'Non-Numeric Data')
