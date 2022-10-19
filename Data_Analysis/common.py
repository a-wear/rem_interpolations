from time import perf_counter, perf_counter_ns
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from data_helper import IDX_X, IDX_Y, FIELD_NAMES
from data_helper import SAMPLES_PER_POINT, DISTANCE_X, DISTANCE_Y

# Set Matplotlib options
#matplotlib.use('Agg')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times New Roman"


def create_radiomap(x, y, z, p, width, height, point, xlim, ylim, zlim, xticks, yticks, results_dir, savename, ip=None, cm='viridis'):
    '''Function to create countour radio map of 2D space'''
    fig, ax = plt.subplots(figsize=(width, height))
    fig.tight_layout()
    ax.set_aspect('equal', 'box')
    surf = ax.contourf(x, y, z, cmap=cm)
    scat1 = ax.scatter([point[1]], [point[2]], color='red', zorder=5)
    scat2 = ax.scatter(x, y, p, color='darkorange', marker='x', zorder=7)
    if ip is not None:
        scat3 = ax.scatter(x, y, ip, color='darkred', marker='x', zorder=7)
    ax.annotate(point[0], (point[1]+0.15, point[2]), zorder=6)
    ax.set_xlabel('Distance X [m]')
    ax.set_ylabel('Distance Y [m]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True)
    cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel('RSSI [dBm]')
    fig.savefig(os.path.join(results_dir, savename))


def interpolate_radiomap(data, interpolation_type):
    '''Function for interpolation of NaN points in 2D array'''
    # Create meshgrid and data mask
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    array = np.ma.masked_invalid(data)
    xx, yy = np.meshgrid(x, y)

    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    return interpolate.griddata((x1, y1), newarr.ravel(),
                                (xx, yy), method=interpolation_type)


def get_gaussian_regression_data(base_data_location=None, X_train=None, y_train=None, distances_x=None, distances_y=None, kernel=None, np_radiomap=True, split=0.3, remove=None):
    '''Train Gaussian Regression process on base data and return grids of predictions'''
    # Create mesh grid
    xx, yy = np.meshgrid(distances_x, distances_y)

    # Get all location coordinates
    coords = np.dstack([xx, yy]).reshape(-1, 2)

    # Data loading non-interpolated data
    # y_train, _, X_train, _ = load_data(base_data_location, split=split_input)
    
    if isinstance(base_data_location, str) and X_train is None and y_train is None:
        # Load all base data
        if split != 0:
            y_train, _, X_train, _ = load_data(base_data_location, test_size=split, split=True)
        else:
            data = load_data(base_data_location, split=False)

        if remove is not None:
            data = pd.concat([data, remove, remove], axis=0).drop_duplicates(keep=False)
            X_train = data[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']]
            y_train = data[['X', 'Y']]
    elif isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        pass
    else:
        return -1

    # Creation of gaussian process regressor
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(y_train.values, X_train.values)

    # Get RSSI for all locations
    rssis = gp.predict(coords)

    out = pd.DataFrame(np.hstack([coords, rssis]), columns=FIELD_NAMES)

    if np_radiomap:
        # Allocate arrays for noninterpolated data
        S1n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S2n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S3n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S4n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S5n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))

        S1n = out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 0].T.reshape(xx.shape[0], xx.shape[1])
        S2n = out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 1].T.reshape(xx.shape[0], xx.shape[1])
        S3n = out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 2].T.reshape(xx.shape[0], xx.shape[1])
        S4n = out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 3].T.reshape(xx.shape[0], xx.shape[1])
        S5n = out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 4].T.reshape(xx.shape[0], xx.shape[1])
        
        return S1n, S2n, S3n, S4n, S5n
    else:
        return [out[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']],
                out[['X', 'Y']]]


def get_per_point_gaussian_regression_data(base_data_location=None, X_train=None, y_train=None, distances_x=None, distances_y=None, kernel=None, np_radiomap=False, split=0.3, remove=None):
    '''Get data through Gaussian Regression'''
    # Create meshgrid
    xx, yy = np.meshgrid(distances_x, distances_y)

    # Get all location coordinates
    coords = np.dstack([xx, yy]).reshape(-1, 2)

    if isinstance(base_data_location, str) and X_train is None and y_train is None:
        # Load all base data
        if split != 0:
            X_train, _, y_train, _ = load_data(base_data_location, test_size=split, split=True)
        else:
            data = load_data(base_data_location, split=False)

        if remove is not None:
            data = pd.concat([data, remove, remove], axis=0).drop_duplicates(keep=False)
            X_train = data[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']]
            y_train = data[['X', 'Y']]
    elif isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        pass
    else:
        return -1

    data = pd.concat([y_train, X_train], axis=1)

    # dfs = [pd.DataFrame(columns=FIELD_NAMES)] * SAMPLES_PER_POINT
    dfs = [data[data.index == i] for i in range(0, SAMPLES_PER_POINT)]

    # Create empty data list 
    out = []

    # Train all gaussian regression functions and return output data
    for df in dfs:
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(df[['X', 'Y']].values, 
               df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values)
        out.append(pd.DataFrame(np.hstack((coords, gp.predict(coords))),
                                columns=FIELD_NAMES))
    
    # Return either numpy 3D array or dataframe with all values
    if np_radiomap:
        # Allocate arrays for noninterpolated data
        S1n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S2n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S3n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S4n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))
        S5n = np.zeros((SAMPLES_PER_POINT, xx.shape[0], xx.shape[1]))

        for idx in range(0, SAMPLES_PER_POINT):
            S1n[idx] = out[idx][['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 0].T.reshape(xx.shape[0], xx.shape[1])
            S2n[idx] = out[idx][['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 1].T.reshape(xx.shape[0], xx.shape[1])
            S3n[idx] = out[idx][['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 2].T.reshape(xx.shape[0], xx.shape[1])
            S4n[idx] = out[idx][['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 3].T.reshape(xx.shape[0], xx.shape[1])
            S5n[idx] = out[idx][['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']].values[:, 4].T.reshape(xx.shape[0], xx.shape[1])
        
        return S1n, S2n, S3n, S4n, S5n
    else:
        # Create and return a dataframe
        ouput_df = pd.concat(out)

        return [ouput_df[['RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']],
                ouput_df[['X', 'Y']]]


def load_data(location, test_size=0.3, split=True):
    '''Data loader function with train-test split option'''
    dfs = []
    for dist_y in IDX_Y:
        for dist_x in IDX_X:
            fname = '{0}\\{1}{2}.csv'.format(location, dist_x, dist_y)

            if os.path.isfile(fname):
                df = pd.read_csv(fname, header=0)
                dfs.append(df)

    df = pd.concat(dfs)

    if split:
        # Data split
        return train_test_split(df.drop(axis=1, labels=['X', 'Y']),
                                df.loc[:, 'X':'Y'], test_size=test_size,
                                random_state=0)
    else:
        return df


def load_data_np_arrays(location, missing_value=np.nan):
    '''Load data and split them by station into 5 arrays'''
    # Data loading
    df = load_data(location, split=False)

    # Allocate arrays for noninterpolated data
    S1n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
    S1n[S1n == 0] = missing_value
    S2n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
    S2n[S2n == 0] = missing_value
    S3n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
    S3n[S3n == 0] = missing_value
    S4n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
    S4n[S4n == 0] = missing_value
    S5n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
    S5n[S5n == 0] = missing_value

    # Split data
    for idx in IDX_X:
        for idy in IDX_Y:
            temp = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_1'])
            if np.any(temp):
                S1n[:, IDX_X[idx], IDX_Y[idy]] = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_1'])
                S2n[:, IDX_X[idx], IDX_Y[idy]] = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_2'])
                S3n[:, IDX_X[idx], IDX_Y[idy]] = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_3'])
                S4n[:, IDX_X[idx], IDX_Y[idy]] = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_4'])
                S5n[:, IDX_X[idx], IDX_Y[idy]] = np.array(df.loc[(df['X'] == DISTANCE_X[idx]) & (df['Y'] == DISTANCE_Y[idy])]['RSSI_5'])

    return S1n, S2n, S3n, S4n, S5n


def timeit(function, *args, ns=False, repeat=1):
    '''Helper function for measuring function execution time'''
    times = []

    if ns:
        timer = perf_counter_ns
    else:
        timer = perf_counter

    for _ in range(0, repeat):
        t_start = timer()
        out = function(*args)
        t_stop = timer()
        times.append(t_stop - t_start)
    return out, times


def timeit_dataframe(function, dataframe):
    '''Helper function for measuring function execution time per row of dataframe'''
    times = []
    out= []
    for i in range(0, len(dataframe)):
        t_start = perf_counter()
        out.append(function(dataframe[i:i+1])[0])
        t_stop = perf_counter()
        times.append(t_stop - t_start)
    return out, times
