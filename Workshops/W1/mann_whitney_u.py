import numpy as np
import csv
import io
from scipy.stats import mannwhitneyu

def mann_whitney_u_test(data1, data2, alternative='two-sided'):
    """
    Performs the Mann-Whitney U test on two independent samples.

    The Mann-Whitney U test (also known as the Wilcoxon rank-sum test)
    is a non-parametric test used to determine whether two independent
    samples are from the same distribution. It is often used as a
    non-parametric alternative to the independent samples t-test when
    the assumption of normality is violated.

    Parameters:
    data1 (array-like): The first sample of data.
    data2 (array-like): The second sample of data.
    alternative (str, optional): Defines the alternative hypothesis.
        'two-sided': The distributions are not equal. (Default)
        'less': The distribution of the first sample is stochastically
                less than that of the second.
        'greater': The distribution of the first sample is stochastically
                   greater than that of the second.

    Returns:
    tuple: A tuple containing the U-statistic and the p-value.
    """
    # Convert input to numpy arrays for consistency and handling
    sample1 = np.asarray(data1)
    sample2 = np.asarray(data2)

    # Check for valid input
    if sample1.size < 1 or sample2.size < 1:
        raise ValueError("Both samples must contain at least one data point.")

    # Perform the Mann-Whitney U test using scipy's implementation
    u_statistic, p_value = mannwhitneyu(sample1, sample2, alternative=alternative)

    return u_statistic, p_value

def read_data_from_csv(file_path, col_a_name='A', col_b_name='B'):
	data_a = []
	data_b = []

	try:
		with open(file_path, 'r') as file:
			reader = csv.reader(file)

			header = next(reader)
			try:
				idx_a = header.index(col_a_name)
				idx_b = header.index(col_b_name)
			except ValueError as e:
				raise ValueError(f"Missing required column in CSV header: {e}")

			for row in reader:
				try:
					data_a.append(float(row[idx_a]))
					data_b.append(float(row[idx_b]))
				except (ValueError, IndexError):
					continue
	except FileNotFoundError:
		raise FileNotFoundError(f"Error: the file '{file_path}' was not found.")

	return data_a, data_b

if __name__ == '__main__':
    csv_file_path = '/Users/kem0149/Library/CloudStorage/OneDrive-AuburnUniversity/Desktop/Software Quality Assurance/W1/perf-data.csv'
    
    try:
        data_a, data_b = read_data_from_csv(csv_file_path)
        
        # Perform the test on the extracted data
        u_stat, p_val = mann_whitney_u_test(data_a, data_b)
        
        print(f"Mann-Whitney U test on data from '{csv_file_path}':")
        print(f"U statistic: {u_stat:.2f}")
        print(f"P-value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("Conclusion: The difference is statistically significant. We reject the null hypothesis.")
        else:
            print("Conclusion: The difference is not statistically significant. We fail to reject the null hypothesis.")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
