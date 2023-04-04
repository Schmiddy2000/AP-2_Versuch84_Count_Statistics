import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, norm, skew, kurtosis
from scipy.optimize import curve_fit


t = 60

V = np.array([300, 450, 500, 520, 523, 528, 540, 550, 563, 567, 576, 587, 600, 605, 615, 624, 631, 644, 650])
X = np.array([0, 0, 0, 33, 83, 92, 83, 92, 77, 101, 112, 102, 87, 100, 110, 115, 97, 104, 98])

R = X / t
del_R = np.sqrt(X) / t

X1 = np.array([3, 2, 3, 2, 0, 1, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 3, 3, 1, 1, 0, 2, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0,
               2, 0, 5, 0, 3, 1, 3, 1, 3, 1, 1, 1, 3, 2, 0, 2, 1, 3, 2, 2, 1, 0, 3, 1, 5, 4, 0, 3, 1, 0, 1, 1,
               1, 3, 3, 4, 2, 0, 0, 1, 0, 1, 2, 2, 0, 4, 1, 1, 1, 0, 0, 2, 3, 3, 1, 3, 1, 1, 4, 0, 2, 0, 1, 3,
               2, 2, 3, 1])

X2 = np.array([4, 3, 5, 4, 1, 4, 4, 6, 4, 4, 6, 3, 5, 4, 3, 2, 2, 0, 4, 4, 4, 4, 4, 5, 3, 4, 5, 2, 3, 2, 1, 3,
               1, 0, 4, 3, 1, 0, 3, 6, 2, 3, 6, 3, 0, 4, 3, 4, 7, 5, 0, 6, 5, 4, 6, 1, 2, 4, 1, 6, 4, 3, 4, 3,
               3, 2, 2, 5, 1, 6, 4, 2, 1, 3, 3, 0, 1, 3, 4, 2, 4, 5, 2, 2, 2, 3, 4, 0, 5, 2, 2, 5, 1, 2, 6, 1,
               2, 5, 4, 0])

X4 = np.array([12, 10, 3, 12, 6, 6, 7, 9, 13, 7, 4, 7, 10, 5, 9, 5, 8, 7, 11, 6, 11, 9, 6, 4, 10, 8, 7, 7, 8, 7,
               7, 7, 3, 10, 5, 11, 5, 8, 4, 10, 6, 6, 4, 7, 4, 9, 6, 6, 5, 10, 4, 7, 6, 10, 5, 4, 4, 6, 7, 11, 3,
               6, 5, 8, 7, 11, 3, 8, 7, 6, 1, 6, 11, 6, 11, 10, 5, 6, 9, 12, 6, 11, 4, 8, 7, 7, 10, 4, 7, 6, 4,
               9, 9, 17, 10, 10, 10, 3, 7, 9])


def r(x): return round(x, 5)


def plotVT1():
    plt.scatter(V, R, marker='x')
    plt.errorbar(V, R, yerr=del_R, fmt='none', capsize=3, ecolor='black')

    return None


def lin_model(x, a, c):
    return a * x + c


def linReg(x_array, y_array, y_errors, print_output=True):
    popt, pcov = curve_fit(lin_model, x_array, y_array, sigma=y_errors)

    a, c = popt
    a_err, c_err = np.sqrt(np.diag(pcov))

    if print_output:
        print('a =', r(a), '+-', r(a_err), '\nc =', r(c), '+-', r(c_err))

    return a, a_err, c, c_err


def plotX(x_array, do_lin_reg = False):
    if min(x_array) == 0:
        x_array = x_array + 1e-6
        adjusted_x_array = True
    else:
        adjusted_x_array = False

    x_axis = np.arange(0, len(x_array))
    x_array_errors = np.sqrt(x_array) / getTimeInterval(x_array)
    plt.scatter(x_axis, x_array, lw=1.5, marker='x')

    if do_lin_reg:
        if adjusted_x_array:
            best_a, best_a_err, best_c, best_c_err = linReg(x_axis, x_array, np.zeros(len(x_array)) + 1e-2)
        else:
            best_a, best_a_err, best_c, best_c_err = linReg(x_axis, x_array, x_array_errors)

        upper = lin_model(x_axis, best_a + best_a_err, best_c - best_c_err)
        lower = lin_model(x_axis, best_a - best_a_err, best_c + best_c_err)
        plt.fill_between(x_axis, upper, lower, where=upper >= lower, interpolate=True, color='pink', alpha=0.5)
        plt.fill_between(x_axis, upper, lower, where=upper < lower, interpolate=True, color='pink', alpha=0.5,
                         label='Konfidenzband')

        plt.plot(x_axis, lin_model(x_axis, best_a, best_c), ls='--', c='black')
    
        plt.legend()

    return None


def getTimeInterval(data_set):

    # Determine measurement time
    if max(data_set) == 5:
        time_interval = 1
    elif max(data_set) == 7:
        time_interval = 2
    else:
        time_interval = 4

    # return the correct error for each value
    return time_interval     # np.sqrt(data_set) / time_interval


def tiltAndBulk(data_set):
    mu = np.mean(data_set)
    coefficient = 1 / len(data_set)
    empirical_std = np.sqrt(1 / (len(data_set) - 1) * sum((data_set - mu) ** 2))
    my_tilt_sum = sum(((data_set - mu) / empirical_std) ** 3)
    my_bulge_sum = sum(((data_set - mu) / empirical_std) ** 4)

    my_tilt = coefficient * my_tilt_sum
    my_bulge = coefficient * my_bulge_sum

    return my_tilt, my_bulge - 3


def plotHistogram(data_set, print_output = True):
    mu = np.mean(data_set)
    var = np.var(data_set)
    sigma = np.std(data_set)

    # Creates a lin-space that accounts for the range of x_values and the width of the bars
    x_range = np.linspace(min(data_set.any(), 0) - 0.5, max(data_set) + 0.5, 100)
    gaussian_pdf = norm.pdf(x_range, mu, sigma)

    time_interval = getTimeInterval(data_set)

    x_values = np.arange(0, np.max(data_set) + 1)  # Generate x values from 0 to the maximum value in the data set
    poisson_pmf = poisson.pmf(x_values, mu)  # Compute the Poisson PMF for the given x values and mu

    plt.stem(x_values, poisson_pmf, markerfmt='.', basefmt=" ", linefmt='C0-', label='Poisson PMF')
    plt.hist(data_set, bins=np.arange(np.max(data_set) + 2) - 0.5, density=True, alpha=0.5, label='Data histogram')
    plt.plot(x_range, gaussian_pdf, label='Gaussian PDF', color='red')
    plt.title('Histogramm der Messungen mit $t$ = 4s', fontsize=16)
    plt.xlabel('Anzahl der Detektionen', fontsize=12)
    plt.ylabel('Wahrscheinlichkeit in [%]', fontsize=12)
    plt.legend()

    if print_output:

        tilt, bulge = tiltAndBulk(data_set)

        print('Messung mit t = ' + str(time_interval) + ':')
        print('Mean =', r(mu), '+-', r(sigma))
        print('The variance is var =', r(var))
        print('Empirical tilt:', r(tilt), '\nEmpirical bulge:', r(bulge))
        print('Built in functions:\nSkewness =', skew(data_set), 'and kurtosis =', kurtosis(data_set))
        print('For Poisson:\nVar =', r(mu), 'Skewness =', r(1 / np.sqrt(mu)), 'and kurtosis =', r(1 / mu))

    return None


def barCharWithErrorbars(data_set):
    # Compute the histogram data using NumPy
    bin_edges = np.arange(np.max(data_set) + 2) - 0.5
    hist_data, _ = np.histogram(data_set, bins=bin_edges)

    # Calculate the errors for each bin (e.g., using the square root of the counts as a simple example)
    errors = np.sqrt(hist_data)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram with error bars using plt.bar
    plt.bar(bin_centers, hist_data, width=1, yerr=errors, capsize=3, alpha=0.5)

    plt.xlabel('Data values')
    plt.ylabel('Percentage')

    return None


plt.figure(figsize=(12, 5))


# plotHistogram(X4)

plotX(X4, True)

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1, wspace=0)
# plt.savefig('Histogramm_t=4.png', dpi=300)
plt.show()


# Messung mit t = 1:
# Mean = 1.59 +- 1.2255202976695245
# The squared std = 1.5019 whereas the variance is: 1.5019

# Messung mit t = 2:
# Mean = 3.15 +- 1.7226433176952218
# The squared std = 2.967500000000001 whereas the variance is: 2.9675000000000007

# Messung mit t = 4:
# Mean = 7.3 +- 2.72213151776324
# The squared std = 7.41 whereas the variance is: 7.41
