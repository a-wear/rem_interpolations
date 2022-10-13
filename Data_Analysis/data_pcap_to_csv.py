import csv
import os
from pathlib import Path

import numpy as np
from scapy.all import rdpcap

from common import interpolate_radiomap
from data_helper import IDX_X, IDX_Y, DISTANCE_X, DISTANCE_Y, FIELD_NAMES
from data_helper import SAMPLES_PER_POINT, INTERPOLATION, DATASET_LOCATION
from data_helper import COMBINED_DATASET_LOCATION, COMBINED_INTERPOLATED_DATASET_LOCATION
from data_helper import DATASET_EVAL_LOCATION, COMBINED_DATASET_EVAL_LOCATION

# Allocate arrays for noninterpolated data
S1n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S1n[S1n == 0] = np.nan
S2n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S2n[S2n == 0] = np.nan
S3n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S3n[S3n == 0] = np.nan
S4n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S4n[S4n == 0] = np.nan
S5n = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S5n[S5n == 0] = np.nan

# Get folder and filenames
dirs = [d for d in os.listdir(DATASET_LOCATION) if os.path.isdir(os.path.join(DATASET_LOCATION, d))]
filenames = os.listdir(os.path.join(DATASET_LOCATION, dirs[0]))

# Create output data if they do not exist
os.makedirs(COMBINED_DATASET_LOCATION, exist_ok=True)
os.makedirs(COMBINED_INTERPOLATED_DATASET_LOCATION, exist_ok=True)

# Load data, combine them and save them
for filename in filenames:
    probes_1 = rdpcap(os.path.join(DATASET_LOCATION, dirs[0], filename))
    probes_2 = rdpcap(os.path.join(DATASET_LOCATION, dirs[1], filename))
    probes_3 = rdpcap(os.path.join(DATASET_LOCATION, dirs[2], filename))
    probes_4 = rdpcap(os.path.join(DATASET_LOCATION, dirs[3], filename))
    probes_5 = rdpcap(os.path.join(DATASET_LOCATION, dirs[4], filename))

    # Get all sequence numbers
    sc1 = [p.payload.SC >> 4 for p in probes_1]
    sc2 = [p.payload.SC >> 4 for p in probes_2]
    sc3 = [p.payload.SC >> 4 for p in probes_3]
    sc4 = [p.payload.SC >> 4 for p in probes_4]
    sc5 = [p.payload.SC >> 4 for p in probes_5]

    # Recognize all similar sequence numbers
    same_sc = [sc for sc in sc1 if sc in sc2 and sc in sc3 and sc in sc4 and sc in sc5]

    # Select only probes with the same sequence numbers
    probes_1 = [p for p in probes_1 if p.payload.SC >> 4 in same_sc]
    probes_2 = [p for p in probes_2 if p.payload.SC >> 4 in same_sc]
    probes_3 = [p for p in probes_3 if p.payload.SC >> 4 in same_sc]
    probes_4 = [p for p in probes_4 if p.payload.SC >> 4 in same_sc]
    probes_5 = [p for p in probes_5 if p.payload.SC >> 4 in same_sc]

    # Get X and Y index from filename
    x = IDX_X[Path(filename).stem[0]]
    y = IDX_Y[Path(filename).stem[1:]]

    # Sort the probes
    probes_1.sort(key=lambda x: x.payload.SC >> 4)
    probes_2.sort(key=lambda x: x.payload.SC >> 4)
    probes_3.sort(key=lambda x: x.payload.SC >> 4)
    probes_4.sort(key=lambda x: x.payload.SC >> 4)
    probes_5.sort(key=lambda x: x.payload.SC >> 4)

    # Read RSSI from probe requests
    for idx, (p1, p2, p3, p4, p5) in enumerate(zip(probes_1, probes_2, probes_3, probes_4, probes_5)):
        S1n[idx][x][y] = p1.dBm_AntSignal
        S2n[idx][x][y] = p2.dBm_AntSignal
        S3n[idx][x][y] = p3.dBm_AntSignal
        S4n[idx][x][y] = p4.dBm_AntSignal
        S5n[idx][x][y] = p5.dBm_AntSignal

    # Save noninterpolted data
    output_file = Path(filename).stem + '.csv'
    with open(os.path.join(COMBINED_DATASET_LOCATION, output_file), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for idx in range(0, SAMPLES_PER_POINT):
            writer.writerow({'X': DISTANCE_X[Path(filename).stem[0]],
                             'Y': DISTANCE_Y[Path(filename).stem[1:]],
                             'RSSI_1': S1n[idx][x][y],
                             'RSSI_2': S2n[idx][x][y],
                             'RSSI_3': S3n[idx][x][y],
                             'RSSI_4': S4n[idx][x][y],
                             'RSSI_5': S5n[idx][x][y]})

# Allocate arrays for interpolated data
S1in = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S1in[S1in == 0] = np.nan
S2in = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S2in[S2in == 0] = np.nan
S3in = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S3in[S3in == 0] = np.nan
S4in = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S4in[S4in == 0] = np.nan
S5in = np.zeros((SAMPLES_PER_POINT, len(DISTANCE_X), len(DISTANCE_Y)))
S5in[S5in == 0] = np.nan

# Interpolation of data
for idx in range(0, SAMPLES_PER_POINT):
    S1in[idx] = interpolate_radiomap(S1n[idx], INTERPOLATION)
    S2in[idx] = interpolate_radiomap(S2n[idx], INTERPOLATION)
    S3in[idx] = interpolate_radiomap(S3n[idx], INTERPOLATION)
    S4in[idx] = interpolate_radiomap(S4n[idx], INTERPOLATION)
    S5in[idx] = interpolate_radiomap(S5n[idx], INTERPOLATION)

# Save interpolated data
for idx_x in IDX_X:
    x = IDX_X[idx_x]
    for idx_y in IDX_Y:
        y = IDX_Y[idx_y]

        # Skip file if there are NaN values
        if np.isnan(S1in[0][x][y]):
            continue

        # Save interpolted data
        output_file = idx_x + idx_y + '.csv'
        with open(os.path.join(COMBINED_INTERPOLATED_DATASET_LOCATION, output_file), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
            writer.writeheader()
            for idx in range(0, SAMPLES_PER_POINT):
                if np.isnan(S1in[idx][x][y]):
                    continue
                writer.writerow({'X': DISTANCE_X[idx_x],
                                 'Y': DISTANCE_Y[idx_y],
                                 'RSSI_1': S1in[idx][x][y],
                                 'RSSI_2': S2in[idx][x][y],
                                 'RSSI_3': S3in[idx][x][y],
                                 'RSSI_4': S4in[idx][x][y],
                                 'RSSI_5': S5in[idx][x][y]})


# Get evaluation folder and filenames
dirs = [d for d in os.listdir(DATASET_EVAL_LOCATION) if os.path.isdir(os.path.join(DATASET_EVAL_LOCATION, d))]
filenames = os.listdir(os.path.join(DATASET_EVAL_LOCATION, dirs[0]))

# Create evaluation output data if they do not exist
os.makedirs(COMBINED_DATASET_EVAL_LOCATION, exist_ok=True)

for filename in filenames:
    probes_1 = rdpcap(os.path.join(DATASET_EVAL_LOCATION, dirs[0], filename))
    probes_2 = rdpcap(os.path.join(DATASET_EVAL_LOCATION, dirs[1], filename))
    probes_3 = rdpcap(os.path.join(DATASET_EVAL_LOCATION, dirs[2], filename))
    probes_4 = rdpcap(os.path.join(DATASET_EVAL_LOCATION, dirs[3], filename))
    probes_5 = rdpcap(os.path.join(DATASET_EVAL_LOCATION, dirs[4], filename))

    # Get all sequence numbers
    sc1 = [p.payload.SC >> 4 for p in probes_1]
    sc2 = [p.payload.SC >> 4 for p in probes_2]
    sc3 = [p.payload.SC >> 4 for p in probes_3]
    sc4 = [p.payload.SC >> 4 for p in probes_4]
    sc5 = [p.payload.SC >> 4 for p in probes_5]

    # Recognize all similar sequence numbers
    same_sc = [sc for sc in sc1 if sc in sc2 and sc in sc3 and sc in sc4 and sc in sc5]

    # Select only probes with the same sequence numbers
    probes_1 = [p for p in probes_1 if p.payload.SC >> 4 in same_sc]
    probes_2 = [p for p in probes_2 if p.payload.SC >> 4 in same_sc]
    probes_3 = [p for p in probes_3 if p.payload.SC >> 4 in same_sc]
    probes_4 = [p for p in probes_4 if p.payload.SC >> 4 in same_sc]
    probes_5 = [p for p in probes_5 if p.payload.SC >> 4 in same_sc]

    # Sort the probes
    probes_1.sort(key=lambda x: x.payload.SC >> 4)
    probes_2.sort(key=lambda x: x.payload.SC >> 4)
    probes_3.sort(key=lambda x: x.payload.SC >> 4)
    probes_4.sort(key=lambda x: x.payload.SC >> 4)
    probes_5.sort(key=lambda x: x.payload.SC >> 4)

    S1_rssi = list()
    S2_rssi = list()
    S3_rssi = list()
    S4_rssi = list()
    S5_rssi = list()

    # Read RSSI from probe requests
    for idx, (p1, p2, p3, p4, p5) in enumerate(zip(probes_1, probes_2, probes_3, probes_4, probes_5)):
        S1_rssi.append(p1.dBm_AntSignal)
        S2_rssi.append(p2.dBm_AntSignal)
        S3_rssi.append(p3.dBm_AntSignal)
        S4_rssi.append(p4.dBm_AntSignal)
        S5_rssi.append(p5.dBm_AntSignal)

    # Save noninterpolted data
    output_file = Path(filename).stem + '.csv'
    with open(os.path.join(COMBINED_DATASET_EVAL_LOCATION, output_file), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for idx in range(0, 25):
            writer.writerow({'X': Path(filename).stem.replace('x', '').split('y')[0],
                             'Y': Path(filename).stem.replace('x', '').split('y')[1],
                             'RSSI_1': S1_rssi[idx],
                             'RSSI_2': S2_rssi[idx],
                             'RSSI_3': S3_rssi[idx],
                             'RSSI_4': S4_rssi[idx],
                             'RSSI_5': S5_rssi[idx]})