import os
import matplotlib.pyplot as plt
import numpy as np

from common import load_data_np_arrays
from data_helper import RESULTS_LOCATION
from data_helper import COMBINED_DATASET_LOCATION, COMBINED_INTERPOLATED_DATASET_LOCATION

# Create Results directory if it does not exist
os.makedirs(os.path.join(RESULTS_LOCATION, dir), exist_ok=True)

# Get filenames for noninterpolated data
filenames = os.listdir(COMBINED_DATASET_LOCATION)

# Load data without interpolation
S1n, _, _, _, _ = load_data_np_arrays(COMBINED_DATASET_LOCATION)

# Do a mean of all samples at one coordinate
S1n_mean = np.mean(S1n, axis=0)

# Create data points
Points = S1n_mean.copy()
Points[Points < 0] = 20

# Create meshgrid for plotting contours
X, Y = np.meshgrid(np.arange(0.85, 16, 1), np.arange(0.1, 11, 1))

# Define plot dimensions
width = 8
height = 4.5

# Get filenames for interpolated data
filenames = os.listdir(COMBINED_INTERPOLATED_DATASET_LOCATION)

# Load data with interpolation
S1in, _, _, _, _ = load_data_np_arrays(COMBINED_INTERPOLATED_DATASET_LOCATION)

# Do a mean of all samples at one coordinate
S1in_mean = np.mean(S1in, axis=0)

fig = plt.figure(figsize=(15, 7.2), constrained_layout=True)
ax = fig.add_subplot()
surf = ax.contourf(X, Y, S1in_mean.T, cmap='plasma')
surf = ax.contourf(X, Y, S1n_mean.T, cmap='viridis')
ax.set_xlim([0.85, 15.85])
ax.set_ylim([0.1, 10.1])
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(os.path.join(RESULTS_LOCATION, 'sensors.svg'))
fig.savefig(os.path.join(RESULTS_LOCATION, 'sensors.eps'))
