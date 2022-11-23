from collections import defaultdict
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from data_helper import PLOT_EXECUTION_TIME_LABEL_N, PLOT_K_LABEL
from data_helper import MAX_K, K, PLOT_MEAN_ERROR_LABEL_M
from data_helper import RESULTS_LOCATION, COMBINED_DATASET_LOCATION
from data_helper import COMBINED_INTERPOLATED_DATASET_LOCATION
from data_helper import COMBINED_DATASET_EVAL_LOCATION
from data_helper import HISTOGRAM_BINS, HISTOGRAM_X_LABEL, HISTOGRAM_Y_LABEL
from data_helper import TIMING_REPETITIONS, TIMES_LOCATION, FIELD_NAMES
from common import load_data, timeit, timeit_dataframe, get_gaussian_regression_data
from common import get_per_point_gaussian_regression_data

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"

# Create Results directory if it does not exist
os.makedirs(RESULTS_LOCATION, exist_ok=True)

rows = 2
columns = 7

#
# Create all figures
#
fig1, axes1 = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3), constrained_layout=True, sharex=True, sharey=True)

fig2, axes2 = plt.subplots(1, figsize=(8, 5), constrained_layout=True)
fig3, axes3 = plt.subplots(1, figsize=(8, 5), constrained_layout=True)
fig4, axes4 = plt.subplots(1, figsize=(8, 5), constrained_layout=True)
fig5, axes5 = plt.subplots(1, figsize=(8, 5), constrained_layout=True)

# fig1.suptitle('Histograms of predicted distances by kNN')
# fig2.suptitle('Error Rate K Value')
# fig3.suptitle('Execution speed')
# fig4.suptitle('Mean Error on Execution speed')

#
# Create all configurations
#
data_formats = ('measured',
                'interpolated',
                'gaus_50p_1m',
                'gaus_50p_05m',
                'lin&gaus_50p_1m',
                'lin&gaus_50p_05m',
                'lin&gausSel_50p_1m',
                'lin&gausSel_50p_05m',
                'gaus_1p_1m',
                'gaus_1p_05m',
                'lin&gaus_1p_1m',
                'lin&gaus_1p_05m',
                'lin&gausSel_1p_1m',
                'lin&gausSel_1p_05m'
                )
algorithm = 'brute' #, 'ball_tree', 'kd_tree')

#
# Create output data storing dictionaries
#
error_mean = defaultdict(None)
times = defaultdict(None)
distances = defaultdict(None)
predictions = defaultdict(None)

#
# Data loading
#
X_train = defaultdict(None)
X_test = defaultdict(None)
y_train = defaultdict(None)
y_test = defaultdict(None)

# Create Mesh  grids for 0.5m and 1m data grid
grid_step_05 = 0.5
x_distances_05 = np.arange(-0.15 + grid_step_05, 17.71 - grid_step_05, grid_step_05)
y_distances_05 = np.arange(-0.9 + grid_step_05, 11.76 - grid_step_05, grid_step_05)
X0_5, Y0_5 = np.meshgrid(x_distances_05, y_distances_05)

grid_step_1 = 1
x_distances_1 = np.arange(-0.15 + grid_step_1, 17.71 - grid_step_1, grid_step_1)
y_distances_1 = np.arange(-0.9 + grid_step_1, 11.76 - grid_step_1, grid_step_1)
X1, Y1 = np.meshgrid(x_distances_1, y_distances_1)

# Data loading non-interpolated data 1
X_train['measured'], X_test, y_train['measured'], y_test = load_data(COMBINED_DATASET_LOCATION, 0.1)

# Data loading interpolated data - Remove Test data measured 2
interpolated = load_data(COMBINED_INTERPOLATED_DATASET_LOCATION, split=False)
test_df = pd.concat([interpolated, pd.concat([y_test, X_test], axis=1),
                     pd.concat([y_test, X_test], axis=1)],
                    axis=0).drop_duplicates(keep=False)
X_train['interpolated'] = test_df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']]
y_train['interpolated'] = test_df[['X', 'Y']]

# Get Gaussian regression data points for 1m grid, 50 points per location 3
X_train['gaus_50p_1m'], y_train['gaus_50p_1m'] = get_per_point_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                                                                       distances_x=x_distances_1,
                                                                                       distances_y=y_distances_1,
                                                                                       np_radiomap=False)

# Get Gaussian regression data points for 0.5m grid, 50 points per location 4
X_train['gaus_50p_05m'], y_train['gaus_50p_05m'] = get_per_point_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                                                                          distances_x=x_distances_05,
                                                                                          distances_y=y_distances_05,
                                                                                          np_radiomap=False)
# Get Gaussian regression data points for 1m grid, 50 point per location 5
data_linGauss_50p_1m = get_per_point_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                                 distances_x=x_distances_1,
                                                 distances_y=y_distances_1,
                                                 np_radiomap=False,
                                                 split=0,
                                                 remove=pd.concat([y_test, X_test], axis=1))


X_train['lin&gaus_50p_1m'], y_train['lin&gaus_50p_1m'] = data_linGauss_50p_1m

# Get Gaussian regression data points for 0.5m grid, 50 point per location 6
data_linGauss_50p_05m = get_per_point_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                                 distances_x=x_distances_05,
                                                 distances_y=y_distances_05,
                                                 np_radiomap=False,
                                                 split=0,
                                                 remove=pd.concat([y_test, X_test], axis=1))


X_train['lin&gaus_50p_05m'], y_train['lin&gaus_50p_05m'] = data_linGauss_50p_05m

# Get Gaussian regression data trained on selection points for 1m grid 7
interpolated_2m_grid = list()
measured = pd.concat([y_train['interpolated'], X_train['interpolated']], axis=1)
for x in np.arange(0.85, 17, 2):
    for y in np.arange(0.1, 11, 2):
        interpolated_2m_grid.append(measured[(measured['X'] == x) & (measured['Y'] == y)])

interpolated_2m_grid_df = pd.concat([df for df in interpolated_2m_grid if df.size != 0])

data_gausSel_50p_1m = get_per_point_gaussian_regression_data(X_train=interpolated_2m_grid_df[['RSSI_1',
                                                                                              'RSSI_2',
                                                                                              'RSSI_3',
                                                                                              'RSSI_4',
                                                                                              'RSSI_5']],
                                                             y_train=interpolated_2m_grid_df[['X', 'Y']],
                                                             distances_x=x_distances_1,
                                                             distances_y=y_distances_1,
                                                             np_radiomap=False)

X_train['lin&gausSel_50p_1m'], y_train['lin&gausSel_50p_1m'] = data_gausSel_50p_1m

# Get Gaussian regression data trained on selection points for 1m grid 8
data_gausSel_50p_05m = get_per_point_gaussian_regression_data(X_train=interpolated_2m_grid_df[['RSSI_1',
                                                                                               'RSSI_2',
                                                                                               'RSSI_3',
                                                                                               'RSSI_4',
                                                                                               'RSSI_5']],
                                                              y_train=interpolated_2m_grid_df[['X', 'Y']],
                                                              distances_x=x_distances_05,
                                                              distances_y=y_distances_05,
                                                              np_radiomap= False)

X_train['lin&gausSel_50p_05m'], y_train['lin&gausSel_50p_05m'] = data_gausSel_50p_05m

# Get Gaussian regression data points for 1m grid, 1 point per location 9
X_train['gaus_1p_1m'], y_train['gaus_1p_1m'] = get_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                                                           distances_x=x_distances_1,
                                                                           distances_y=y_distances_1,
                                                                           np_radiomap=False,
                                                                           split=0,
                                          remove=pd.concat([y_test, X_test], axis=1))

# Get Gaussian regression data points for 0.5m grid, 1 point per location 10
X_train['gaus_1p_05m'], y_train['gaus_1p_05m'] = get_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                                                             distances_x=x_distances_05,
                                                                             distances_y=y_distances_05,
                                                                             np_radiomap=False,
                                                                             split=0,
                                          remove=pd.concat([y_test, X_test], axis=1))

# Get Gaussian regression data points for 1m grid, 50 point per location 11
data_linGauss_1p_1m = get_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                          distances_x=x_distances_1,
                                          distances_y=y_distances_1,
                                          np_radiomap=False,
                                          split=0,
                                          remove=pd.concat([y_test, X_test], axis=1))


X_train['lin&gaus_1p_1m'], y_train['lin&gaus_1p_1m'] = data_linGauss_1p_1m

# Get Gaussian regression data points for 0.5m grid, 50 point per location 12
data_linGauss_1p_05m = get_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                           distances_x=x_distances_05,
                                           distances_y=y_distances_05,
                                           np_radiomap=False,
                                           split=0,
                                           remove=pd.concat([y_test, X_test], axis=1))

X_train['lin&gaus_1p_05m'], y_train['lin&gaus_1p_05m'] = data_linGauss_1p_05m

# Get Gaussian regression data trained on selection points for 1m grid, 1 point per location 13
data_gausSel_1p_1m = get_gaussian_regression_data(X_train=interpolated_2m_grid_df[['RSSI_1',
                                                                     'RSSI_2',
                                                                     'RSSI_3',
                                                                     'RSSI_4',
                                                                     'RSSI_5']],
                                                y_train=interpolated_2m_grid_df[['X', 'Y']],
                                                distances_x=x_distances_1,
                                                distances_y=y_distances_1, np_radiomap=False)

X_train['lin&gausSel_1p_1m'], y_train['lin&gausSel_1p_1m'] = data_gausSel_1p_1m

# Get Gaussian regression data trained on selection points for 0.5m grid, 1 point per location 14
data_gausSel_1p_05m = get_gaussian_regression_data(X_train=interpolated_2m_grid_df[['RSSI_1',
                                                                     'RSSI_2',
                                                                     'RSSI_3',
                                                                     'RSSI_4',
                                                                     'RSSI_5']],
                                                y_train=interpolated_2m_grid_df[['X', 'Y']],
                                                distances_x=x_distances_05,
                                                distances_y=y_distances_05, np_radiomap=False)

X_train['lin&gausSel_1p_05m'], y_train['lin&gausSel_1p_05m'] = data_gausSel_1p_05m

# Load Eval data
dfs = []
for fname in os.listdir(COMBINED_DATASET_EVAL_LOCATION):
    if os.path.isfile(os.path.join(COMBINED_DATASET_EVAL_LOCATION, fname)):
        df = pd.read_csv(os.path.join(COMBINED_DATASET_EVAL_LOCATION, fname), header=0)
        dfs.append(df)

df = pd.concat(dfs)

eval_x = df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']]
eval_y = df[['X', 'Y']]

X_test = pd.concat([X_test, eval_x])
y_test = pd.concat([y_test, eval_y])

k_list = dict()

for data_format in data_formats:
    k_list[data_format] = int(np.round(np.sqrt(len(X_train[data_format]))))

#
# Run kNNs in all configurations
#
for data_format in data_formats:
    error_mean[data_format] = list()
    times[data_format] = list()
    distances[data_format] = list()
    predictions[data_format] = list()

    for k in range(1, MAX_K):
        if k > len(X_train[data_format]):
            break

        print('Data format: {}, K: {}           '.format(data_format, k), end='\r')
        # Prepare kNN
        knn = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm)
        knn.fit(X_train[data_format], y_train[data_format])

        # Run timed predictions
        pred, time = timeit(knn.predict, X_test, repeat=TIMING_REPETITIONS)

        # Calculate distances
        dists = np.linalg.norm(pred - y_test.values, axis=1)

        # Calculate mean and median errors
        error_mean[data_format].append(np.mean(dists))
        times[data_format].append(np.array(time))
        distances[data_format].append(dists)
        predictions[data_format].append(pred)

#
# Chart titles
#             
hist_titles = {'measured': 'Measured data\n(50P/RP, 1m grid)',
               'interpolated': 'Interpolated data\n(50P/RP, 1m grid)',
               'gaus_50p_1m': 'GPR - Measured data\n(50P/RP, 1m grid)',
               'gaus_50p_05m': 'GPR - Measured data\n(50P/RP, 0.5m grid)',
               'lin&gaus_50p_1m': 'GPR - Linearly Interpolated data\n(50P/RP, 1m grid)',
               'lin&gaus_50p_05m': 'GPR - Linearly Interpolated data\n(50P/RP, 0.5m grid)',
               'lin&gausSel_50p_1m': 'GPR - Selection of Linearly Interpolated data\n(50P/RP, 1m grid)',
               'lin&gausSel_50p_05m': 'GPR - Selection of Linearly Interpolated data\n(50P/RP, 0.5m grid)',
               'gaus_1p_1m': 'GPR - Measured data\n(1P/RP, 1m grid)',
               'gaus_1p_05m': 'GPR - Measured data\n(1P/RP, 0.5m grid)',
               'lin&gaus_1p_1m': 'GPR - Linearly Interpolated data\n(1P/RP, 1m grid)',
               'lin&gaus_1p_05m': 'GPR - Linearly Interpolated data\n(1P/RP, 0.5m grid)',
               'lin&gausSel_1p_1m': 'GPR - Selection of Linearly Interpolated data\n(1P/RP, 1m grid)',
               'lin&gausSel_1p_05m': 'GPR - Selection of Linearly Interpolated data\n(1P/RP, 0.5m grid)'
               }

#
# Plot all charts
#              
marker_1m = 'o'
marker_05m = '^'

line_style = {'measured': ('#1f77b4', 'solid', 2, 'Measured REM (50P/RP, 1m grid)', marker_1m),
              'interpolated': ('#ff7f0e', 'solid', 1, 'REM with LID (50P/RP, 1m grid)', marker_1m),
              'gaus_50p_1m': ('#2ca02c', 'solid', 1, 'REM by GPR trained on MD  (50P/RP, 1m grid)', marker_1m),
              'gaus_50p_05m': ('#2ca02c', 'dashed', 1, 'REM by GPR trained on MD  (50P/RP, 0.5m grid)', marker_05m),
              'lin&gaus_50p_1m': ('#d62728', 'solid', 1, 'REM by GPR trained on LID (50P/RP, 1m grid)', marker_1m),
              'lin&gaus_50p_05m': ('#d62728', 'dashed', 1, 'REM by GPR trained on LID (50P/RP, 0.5m grid)', marker_05m),
              'lin&gausSel_50p_1m': ('#9467bd', 'solid', 1, 'REM by GPR trained on Selection of LID (50P/RP, 1m grid)', marker_1m),
              'lin&gausSel_50p_05m': ('#9467bd', 'dashed', 1, 'REM by GPR trained on Selection of LID (50P/RP, 0.5m grid)', marker_05m),
              'gaus_1p_1m': ('#8c564b', 'solid', 1, 'REM by GPR trained on MD  (1P/RP, 1m grid)', marker_1m),
              'gaus_1p_05m': ('#8c564b', 'dashed', 1, 'REM by GPR trained on MD  (1P/RP, 0.5m grid)', marker_05m),
              'lin&gaus_1p_1m': ('#e377c2', 'solid', 1, 'REM by GPR trained on LID (1P/RP, 1m grid)', marker_1m),
              'lin&gaus_1p_05m': ('#e377c2', 'dashed', 1, 'REM by GPR trained on LID (1P/RP, 0.5m grid)', marker_05m),
              'lin&gausSel_1p_1m': ('#7f7f7f', 'solid', 1, 'REM by GPR trained on Selection of LID (1P/RP, 1m grid)', marker_1m),
              'lin&gausSel_1p_05m': ('#7f7f7f', 'dashed', 1, 'REM by GPR trained on Selection of LID (1P/RP, 0.5m grid)', marker_05m)}

# plot histograms
inset_axes5_a = zoomed_inset_axes(axes5,
                                2.5, # zoom = 0.5 ''
                                loc='lower left',
                                bbox_to_anchor=(.55, .7),
                                bbox_transform=axes5.transAxes)
inset_axes5_b = zoomed_inset_axes(axes5,
                                6, # zoom = 0.5 ''
                                loc='lower left',
                                bbox_to_anchor=(.21, .035),
                                bbox_transform=axes5.transAxes)

for idx, data_format in enumerate(data_formats):
    row = int(idx / columns)
    column = int(idx % columns)

    # Set titles, limits etc for the plots
    axes1[row, column].set_title(hist_titles[data_format] + ' K={}'.format(k_list[data_format]))
    axes1[row, column].hist(distances[data_format][k_list[data_format]], HISTOGRAM_BINS, density=True)

    sorted_data = np.sort(distances[data_format][k_list[data_format]])
    cdf = 1. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)

    axes5.plot(sorted_data, cdf,
               color=line_style[data_format][0],
               linestyle=line_style[data_format][1],
               linewidth=line_style[data_format][2],
               label=line_style[data_format][3])
    inset_axes5_a.plot(sorted_data, cdf,
                     color=line_style[data_format][0],
                     linestyle=line_style[data_format][1],
                     linewidth=line_style[data_format][2],
                     label=line_style[data_format][3])
    inset_axes5_b.plot(sorted_data, cdf,
                     color=line_style[data_format][0],
                     linestyle=line_style[data_format][1],
                     linewidth=line_style[data_format][2],
                     label=line_style[data_format][3])

    axes1[row, column].grid()
    if row == rows - 1:
        axes1[row, column].set_xlabel(HISTOGRAM_X_LABEL)
    if column == 0:
        axes1[row, column].set_ylabel(HISTOGRAM_Y_LABEL)

axes5.legend(loc='lower right')
axes5.set_xlim([0, np.max([np.max(distances[x]) for x in distances])])
axes5.set_ylim([0, 1])
axes5.grid()
axes5.set_xlabel(HISTOGRAM_X_LABEL)
axes5.set_ylabel('Probability [-]')

inset_axes5_a.set_ylim([0.9, 1.0])
inset_axes5_a.set_xlim([3.8, 6.6])
inset_axes5_a.set_xticks(np.linspace(3.8, 6.6, 8))
inset_axes5_a.grid()

inset_axes5_b.set_ylim([0.45, 0.55])
inset_axes5_b.set_yticks(np.linspace(0.45, 0.55, 3))
inset_axes5_b.set_xlim([1.8, 2.4])
inset_axes5_b.grid()

mark_inset(axes5, inset_axes5_a, loc1=3, loc2=1, fc="none", ec="0.5")
mark_inset(axes5, inset_axes5_b, loc1=3, loc2=2, fc="none", ec="0.5")

inset_axes2 = zoomed_inset_axes(axes2,
                                2.4, # zoom = 0.5 ''
                                loc='upper left',
                                bbox_to_anchor=(.035, 1),
                                bbox_transform=axes2.transAxes)

# Plot error and performance
for error, time, style in zip(list(error_mean.values()), [np.mean(np.array(x), axis=1) for x in np.array(list(times.values()))], line_style):
    axes2.plot(range(1, len(error) + 1), error,
                color=line_style[style][0],
                linestyle=line_style[style][1],
                linewidth=line_style[style][2],
                label=line_style[style][3])
    inset_axes2.plot(range(1, len(error) + 1), error,
                     color=line_style[style][0],
                     linestyle=line_style[style][1],
                     linewidth=line_style[style][2],
                     label=line_style[style][3])
    axes3.plot(range(1, len(time) + 1), time,
               color=line_style[style][0],
               linestyle=line_style[style][1],
               linewidth=line_style[style][2],
               label=line_style[style][3])
    axes4.scatter(time, error,
                  color=line_style[style][0],
                  linestyle=line_style[style][1],
                  label=line_style[style][3],
                  marker=line_style[style][4],
                  alpha=0.33,
                  edgecolor='none')    

# Set plot 2 parameters
axes2.set_xlim([0, MAX_K - 1])
axes2.set_xticks(np.arange(0, 251, 25))
axes2.set_ylim(bottom=2.2)
axes2.set_xlabel(PLOT_K_LABEL)
axes2.set_ylabel(PLOT_MEAN_ERROR_LABEL_M)
axes2.grid()
axes2.legend(loc='upper right')

inset_axes2.set_xlim([0, 40])
inset_axes2.set_ylim([2.2, 2.8])
inset_axes2.grid()

mark_inset(axes2, inset_axes2, loc1=2, loc2=4, fc="none", ec="0.5")

# Set plot 3 parameters
axes3.set_xlim([0, MAX_K - 1])
axes3.set_xlabel(PLOT_K_LABEL)
axes3.set_ylabel(PLOT_EXECUTION_TIME_LABEL_N)
axes3.grid()
axes3.legend(bbox_to_anchor=(1, 0.88), loc='upper right')

# Set plot 4 parameters
axes4.set_ylabel(PLOT_MEAN_ERROR_LABEL_M)
axes4.set_xlabel(PLOT_EXECUTION_TIME_LABEL_N)
axes4.legend(ncol=1, bbox_to_anchor=(0.3, 1), loc='upper left')
axes4.grid()

# Set size of text in plots 2, 3 and 4
for item in ([axes2.xaxis.label, axes2.yaxis.label,
              axes3.xaxis.label, axes3.yaxis.label,
              axes4.xaxis.label, axes4.yaxis.label,
              axes5.xaxis.label, axes5.yaxis.label] +
             axes2.get_xticklabels() + axes2.get_yticklabels() +
             axes3.get_xticklabels() + axes3.get_yticklabels() +
             axes4.get_xticklabels() + axes4.get_yticklabels() +
             axes5.get_xticklabels() + axes5.get_yticklabels()):
    item.set_fontsize(14)

plt.show()

# Generate error heatmaps
X, Y = np.meshgrid(np.arange(0.85, 16, 1), np.arange(0.1, 11, 1))

error_maps = defaultdict(None)

for data_format in data_formats:
    error_maps[data_format] = np.zeros_like(X)
    error_maps[data_format][:] = np.nan
    
    temp_test = y_test.copy(deep=True)
    temp_test.insert(2, data_format, distances[data_format][k_list[data_format]])

    for x in range(0, X.shape[1]):
        for y in range(0, Y.shape[0]):
            error_maps[data_format][y, x] = np.mean(temp_test[data_format][(y_test['X'] == X[y,x]) & (y_test['Y'] == Y[y,x])].values)

for data_format in data_formats:
    y_test.insert(2, data_format, distances[data_format][k_list[data_format]])
    
    fig, ax = plt.subplots(figsize=(6, 3.375))
    fig.tight_layout()

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Distance X [m]')
    ax.set_ylabel('Distance Y [m]')
    ax.set_xticks(np.arange(0, 17, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xlim([0, 16.71])
    ax.set_ylim([0, 10.76])

    # Limits selected from 3 heatmaps included in the paper
    heatmap = ax.pcolormesh(X, Y, error_maps[data_format], cmap='jet', 
                            vmin=np.min([np.nanmin(error_maps['measured']),
                                         np.nanmin(error_maps['lin&gaus_50p_1m']),
                                         np.nanmin(error_maps['lin&gaus_1p_1m'])]),
                            vmax=np.max([np.nanmax(error_maps['measured']),
                                         np.nanmax(error_maps['lin&gaus_50p_1m']),
                                         np.nanmax(error_maps['lin&gaus_1p_1m'])]))
    cbar = fig.colorbar(heatmap)
    cbar.ax.set_ylabel(PLOT_MEAN_ERROR_LABEL_M)

    fig.savefig(os.path.join(RESULTS_LOCATION, 'error_heatmap_{}.eps'.format(data_format)))
    fig.savefig(os.path.join(RESULTS_LOCATION, 'error_heatmap_{}.svg'.format(data_format)))

plt.show()

fig1.savefig(os.path.join(RESULTS_LOCATION, 'kNN_histograms.svg'))
fig2.savefig(os.path.join(RESULTS_LOCATION, 'kNN_Error_dependency_on_K.svg'))
fig3.savefig(os.path.join(RESULTS_LOCATION, 'kNN_times.svg'))
fig4.savefig(os.path.join(RESULTS_LOCATION, 'kNN_error_on_time.svg'))
fig5.savefig(os.path.join(RESULTS_LOCATION, 'kNN_cdf.svg'))
fig1.savefig(os.path.join(RESULTS_LOCATION, 'kNN_histograms.eps'))
fig2.savefig(os.path.join(RESULTS_LOCATION, 'kNN_Error_dependency_on_K.eps'))
fig3.savefig(os.path.join(RESULTS_LOCATION, 'kNN_times.eps'))
fig4.savefig(os.path.join(RESULTS_LOCATION, 'kNN_error_on_time.eps'))
fig5.savefig(os.path.join(RESULTS_LOCATION, 'kNN_cdf.eps'))
