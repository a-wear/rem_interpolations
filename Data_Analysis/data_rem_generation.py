import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_data
from common import create_radiomap, load_data_np_arrays, get_per_point_gaussian_regression_data
from data_helper import RESULTS_LOCATION
from data_helper import COMBINED_DATASET_LOCATION, COMBINED_INTERPOLATED_DATASET_LOCATION

# Colormap
colormap = 'viridis'

# Common zlim
zlim = [-72, -48]
zticks = np.arange(zlim[0], zlim[1] + 1, 3)

# Create Results directory if it does not exist
os.makedirs(RESULTS_LOCATION, exist_ok=True)

# Get filenames for noninterpolated data
filenames = os.listdir(COMBINED_DATASET_LOCATION)

# Load data without interpolation
S1n, S2n, S3n, S4n, S5n = load_data_np_arrays(COMBINED_DATASET_LOCATION)

# Do a mean of all samples at one coordinate
S1n_mean = np.mean(S1n, axis=0)
S2n_mean = np.mean(S2n, axis=0)
S3n_mean = np.mean(S3n, axis=0)
S4n_mean = np.mean(S4n, axis=0)
S5n_mean = np.mean(S5n, axis=0)

# Create data points
Points = S1n_mean.copy()
Points[Points < 0] = 20

# Create meshgrid for plotting contours
X, Y = np.meshgrid(np.arange(0.85, 16, 1), np.arange(0.1, 11, 1))

# Define plot dimensions
width = 6
height = 3.375

# Plot contours WITHOUT interpolation
create_radiomap(X, Y, S1n_mean.T, Points.T, width, height, 
                ('S1', 3.75, 8.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S1_mean_map.eps', cm=colormap)
create_radiomap(X, Y, S2n_mean.T, Points.T, width, height, 
                ('S2', 13.75, 9.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S2_mean_map.eps', cm=colormap)   
create_radiomap(X, Y, S3n_mean.T, Points.T, width, height, 
                ('S3', 9, 6.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S3_mean_map.eps', cm=colormap)
create_radiomap(X, Y, S4n_mean.T, Points.T, width, height, 
                ('S4', 3.75, 3.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S4_mean_map.eps', cm=colormap)
create_radiomap(X, Y, S5n_mean.T, Points.T, width, height, 
                ('S5', 13.75, 4.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S5_mean_map.eps', cm=colormap)

# Get filenames for interpolated data
filenames = os.listdir(COMBINED_INTERPOLATED_DATASET_LOCATION)

# Load data with interpolation
S1in, S2in, S3in, S4in, S5in = load_data_np_arrays(COMBINED_INTERPOLATED_DATASET_LOCATION)

# Do a mean of all samples at one coordinate
S1in_mean = np.mean(S1in, axis=0)
S2in_mean = np.mean(S2in, axis=0)
S3in_mean = np.mean(S3in, axis=0)
S4in_mean = np.mean(S4in, axis=0)
S5in_mean = np.mean(S5in, axis=0)

# Create data for interpolated scatter
interpolated_points = copy.deepcopy(Points)
interpolated_points[np.isnan(interpolated_points)] = 0
interpolated_points[interpolated_points == 20] = np.nan
interpolated_points[interpolated_points == 0] = 20
interpolated_points[np.isnan(S1in_mean)] = np.nan

# Plot contours WITH interpolation
create_radiomap(X, Y, S1in_mean.T, Points.T, width, height, 
                ('S1', 3.75, 8.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S1_mean_map_interpol.eps',
                interpolated_points.T, colormap)
create_radiomap(X, Y, S2in_mean.T, Points.T, width, height, 
                ('S2', 13.75, 9.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S2_mean_map_interpol.eps',
                interpolated_points.T, colormap)   
create_radiomap(X, Y, S3in_mean.T, Points.T, width, height, 
                ('S3', 9, 6.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S3_mean_map_interpol.eps',
                interpolated_points.T, colormap)
create_radiomap(X, Y, S4in_mean.T, Points.T, width, height, 
                ('S4', 3.75, 3.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S4_mean_map_interpol.eps',
                interpolated_points.T, colormap)
create_radiomap(X, Y, S5in_mean.T, Points.T, width, height, 
                ('S5', 13.75, 4.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S5_mean_map_interpol.eps',
                interpolated_points.T, colormap)

# Create 0.5m meshgrid
grid_step = 0.5
x_distances_05 = np.arange(-0.15, 17.71, grid_step)
y_distances_05 = np.arange(-0.4, 11.76, grid_step)
X0_5, Y0_5 = np.meshgrid(x_distances_05, y_distances_05)

data = get_per_point_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                              distances_x=x_distances_05,
                                              distances_y=y_distances_05,
                                              np_radiomap=True)
S1g, S2g, S3g, S4g, S5g = data

# Create data points
Points_G = np.mean(S1g.copy(), axis=0)
Points_G[Points_G < 0] = 20

fig = plt.figure()
ax = plt.axes(projection='3d')
surf1 = ax.plot_surface(X0_5, Y0_5, np.mean(S1g, axis=0), cmap=colormap, zorder=4, vmin=zlim[0], vmax=zlim[1])

ax.set_xlabel('Distance X [m]')
ax.set_ylabel('Distance Y [m]')
ax.set_zlim(zlim)
ax.set_zticks(zticks)
ax.set_zlabel('RSSI [dBm]')
cbar = fig.colorbar(surf1, pad=0.15, shrink=0.6)
cbar.ax.set_ylabel('RSSI [dBm]')
cbar.ax.set_yticks(zticks)
ax.plot3D([3.75, 3.75], [8.5 , 8.5], [-52, -48], 'red', zorder=5, label='S1')
ax.text(3.75, 8.5, -48, 'S1', color='red', weight='bold')
fig.savefig(os.path.join(RESULTS_LOCATION, 'S1_gauss.eps'))

# Create 1m meshgrid
grid_step = 1
x_distances_1 = np.arange(-0.15, 17.71, grid_step)
y_distances_1 = np.arange(-0.9, 11.76, grid_step)
X1, Y1 = np.meshgrid(x_distances_1, y_distances_1)

data_05 = get_per_point_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                              distances_x=x_distances_05,
                                              distances_y=y_distances_05,
                                              np_radiomap=True)

S1g_05, S2g_05, S3g_05, S4g_05, S5g_05 = data_05

fig = plt.figure()
ax = plt.axes(projection='3d')
surf2 = ax.plot_surface(X0_5, Y0_5, np.mean(S1g_05, axis=0), cmap=colormap, vmin=zlim[0], vmax=zlim[1])
ax.set_xlabel('Distance X [m]')
ax.set_ylabel('Distance Y [m]')
ax.set_zlim(zlim)
ax.set_zticks(zticks)
ax.set_zlabel('RSSI [dBm]')
cbar = fig.colorbar(surf2, pad=0.15, shrink=0.6)
cbar.ax.set_ylabel('RSSI [dBm]')
cbar.ax.set_yticks(zticks)
ax.plot3D([3.75, 3.75], [8.5 , 8.5], [-50, -48], 'red', zorder=5, label='S1')
ax.text(3.75, 8.5, -48, 'S1', color='red', weight='bold')
fig.savefig(os.path.join(RESULTS_LOCATION, 'S1_gauss_interpolated.eps'))

data_1 = get_per_point_gaussian_regression_data(base_data_location=COMBINED_INTERPOLATED_DATASET_LOCATION,
                                              distances_x=x_distances_1,
                                              distances_y=y_distances_1,
                                              np_radiomap=True)

S1g_1, S2g_1, S3g_1, S4g_1, S5g_1 = data_1

create_radiomap(X0_5, Y0_5, np.mean(S1g_05, axis=0), None, width, height, 
                ('S1', 3.75, 8.5), [0, 16.71], [0, 10.76], zlim,
                np.arange(0, 17, 1), np.arange(0, 11, 1),
                RESULTS_LOCATION, 'S1_gauss_interpol_contour.eps',
                None, colormap, contour=False)

fig = plt.figure()
ax = plt.axes(projection='3d')

data_1m = get_per_point_gaussian_regression_data(base_data_location=COMBINED_DATASET_LOCATION,
                                              distances_x=x_distances_1,
                                              distances_y=y_distances_1,
                                              np_radiomap=True)

S1g_1m, S2g_1m, S3g_1m, S4g_1m, S5g_1m = data_1m

surf3 = ax.plot_surface(X1, Y1, np.mean(S1g_1m, axis=0), cmap=colormap, vmin=zlim[0], vmax=zlim[1])

X_train, _, y_train, y_test = load_data(COMBINED_INTERPOLATED_DATASET_LOCATION)

# Create reduced 2m grid size out of measured data
measured_2m_list = list()
measured = pd.concat([y_train, X_train], axis=1)
for x in np.arange(0.85, 17, 2):
    for y in np.arange(0.1, 11, 2):
        measured_2m_list.append(measured[(measured['X'] == x) & (measured['Y'] == y)])

measured_2m = pd.concat([df for df in measured_2m_list if df.size != 0])

gaus2m = get_per_point_gaussian_regression_data(X_train=measured_2m[['RSSI_1',
                                                                     'RSSI_2',
                                                                     'RSSI_3',
                                                                     'RSSI_4',
                                                                     'RSSI_5']],
                                                y_train=measured_2m[['X', 'Y']],
                                                distances_x=x_distances_05,
                                                distances_y=y_distances_05,
                                                np_radiomap=True)


fig = plt.figure()
ax = plt.axes(projection='3d')
surf4 = ax.plot_surface(X0_5, Y0_5, np.mean(gaus2m[0], axis=0), cmap=colormap, vmin=zlim[0], vmax=zlim[1])
ax.set_xlabel('Distance X [m]')
ax.set_ylabel('Distance Y [m]')
ax.set_zlim(zlim)
ax.set_zticks(zticks)
ax.set_zlabel('RSSI [dBm]')
cbar = fig.colorbar(surf4, pad=0.15, shrink=0.6)
cbar.ax.set_ylabel('RSSI [dBm]')
cbar.ax.set_yticks(zticks)
ax.plot3D([3.75, 3.75], [8.5 , 8.5], [-52, -48], 'red', zorder=5, label='S1')
ax.text(3.75, 8.5, -48, 'S1', color='red', weight='bold')
fig.savefig(os.path.join(RESULTS_LOCATION, 'S1_gauss_interpolated_2m.eps'))
plt.show()