import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.optimize import curve_fit

t = 60

V = np.array([300, 450, 500, 520, 523, 528, 540, 550, 563, 567, 576, 587, 600, 605, 615, 624, 631, 644, 650])
X = np.array([0, 0, 0, 33, 83, 92, 83, 92, 77, 101, 112, 102, 87, 100, 110, 115, 97, 104, 98])

R = X / t
del_R = np.sqrt(X) / t


def r(x): return round(x, 3)


def lin_model(x, a, c):
    return a * x + c


def linReg(x_array, y_array, y_errors, print_output=True):
    popt, pcov = curve_fit(lin_model, x_array, y_array, sigma=y_errors)

    a, c = popt
    a_err, c_err = np.sqrt(np.diag(pcov))

    if print_output:
        print('a =', r(a), '+-', r(a_err), '\nc =', r(c), '+-', r(c_err))

    return a, a_err, c, c_err


def createZoomPlot(x_data, y_data, y_data_errors, zoom_x: tuple = None, zoom_y: tuple = None, plot_lin_reg = False,
                   lin_reg_start_index = None):
    # Set the zoomed area
    if zoom_x is None:
        zoom_x = [500, 670]
    if zoom_y is None:
        zoom_y = [1.05, 2.15]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle('Zählrate in Abhängigkeit von der Zählrohrspannung', fontsize=16)

    # Plot data
    ax1.scatter(x_data, y_data, marker='x', label='Messwerte', c='b')
    ax1.errorbar(x_data, y_data, yerr=y_data_errors, fmt='none', capsize=3, elinewidth=0.7, capthick=0.7,
                 ecolor='black', label='Fehler')
    ax1.set_xlabel('Spannung in [V]', fontsize=12)
    ax1.set_ylabel(r'Zählrate $R$ in [s$^{-1}$]', fontsize=12)
    ax1.legend()

    if plot_lin_reg:
        best_a, best_a_err, best_c, best_c_err = linReg(x_data[lin_reg_start_index:], y_data[lin_reg_start_index:],
                                                        y_data_errors[lin_reg_start_index:])

        x_lin = np.linspace(zoom_x[0], zoom_x[1], 100)
        upper = lin_model(x_lin, best_a + best_a_err, best_c - best_c_err)
        lower = lin_model(x_lin, best_a - best_a_err, best_c + best_c_err)
        ax2.fill_between(x_lin, upper, lower, where=upper >= lower, interpolate=True, color='pink', alpha=0.5)
        ax2.fill_between(x_lin, upper, lower, where=upper < lower, interpolate=True, color='pink', alpha=0.5,
                         label='Konfidenzband')
        ax2.plot(x_lin, lin_model(x_lin, best_a, best_c), ls='--', lw=0.7, c='black', label='Ausgleichsgerade')

    ax2.errorbar(x_data, y_data, yerr=y_data_errors, fmt='none', capsize=3, elinewidth=0.7, capthick=0.7,
                 ecolor='black')
    ax2.scatter(x_data, y_data, marker='x', c='b')
    ax2.set_xlabel('Spannung in [V]', fontsize=12)
    ax2.legend()

    # Set the zoomed area limits in the second subplot
    ax2.set_xlim(zoom_x)
    ax2.set_ylim(zoom_y)

    # Draw a rectangle in the first subplot
    ax1.add_patch(plt.Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                                edgecolor='red', linewidth=0.75, facecolor='none', alpha=0.5))

    # Create connection lines between subplots
    con1 = ConnectionPatch(xyA=(zoom_x[1], zoom_y[0]), xyB=(zoom_x[0], zoom_y[0]),
                           coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="red", ls="--", lw=0.7,
                           alpha=0.5)
    con2 = ConnectionPatch(xyA=(zoom_x[1], zoom_y[1]), xyB=(zoom_x[0], zoom_y[1]),
                           coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="red", ls="--", lw=0.7,
                           alpha=0.5)

    # Add the connection lines to the subplots
    ax1.add_artist(con1)
    ax1.add_artist(con2)

    plt.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.1, wspace=0.1)

    for spine in ax2.spines.values():
        spine.set_edgecolor('red')
        spine.set_alpha(0.5)
        spine.set_linewidth(0.7)

    return None


createZoomPlot(V, R, del_R, plot_lin_reg=True, lin_reg_start_index=4)

plt.savefig('Count_rate_gegen_Detektorspannung.png', dpi=300)
plt.show()
