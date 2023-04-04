import numpy as np
import data as d

# Replace these with the actual data sets for the 1s, 2s, and 4s measurements
data_set_1s = d.X1
data_set_2s = d.X2
data_set_4s = d.X4

# Calculate the mean count for each time interval
mean_count_1s = np.mean(data_set_1s)
mean_count_2s = np.mean(data_set_2s)
mean_count_4s = np.mean(data_set_4s)

# Calculate the count rate for each time interval
count_rate_1s = mean_count_1s / 1
count_rate_2s = mean_count_2s / 2
count_rate_4s = mean_count_4s / 4

# Calculate the error for each count rate
error_1s = (np.std(data_set_1s) / np.sqrt(len(data_set_1s))) / 1
error_2s = (np.std(data_set_2s) / np.sqrt(len(data_set_2s))) / 2
error_4s = (np.std(data_set_4s) / np.sqrt(len(data_set_4s))) / 4

print("Background radiation rates (cps) and errors:")
print("1s: {:.2f} ± {:.2f}".format(count_rate_1s, error_1s))
print("2s: {:.2f} ± {:.2f}".format(count_rate_2s, error_2s))
print("4s: {:.2f} ± {:.2f}".format(count_rate_4s, error_4s))
