DATASET_LOCATION = 'Data\\'
DATASET_EVAL_LOCATION = 'Data_Eval\\'
COMBINED_DATASET_LOCATION = 'Processed_Data\\Combined'
COMBINED_DATASET_EVAL_LOCATION = 'Processed_Data\\Combined_Eval'
COMBINED_INTERPOLATED_DATASET_LOCATION = 'Processed_Data\\Combined_Interpolated'
RESULTS_LOCATION = 'Results\\'
TIMES_LOCATION = 'Processed_Data\\Times'

INTERPOLATION = 'linear'

HISTOGRAM_BINS = 25

TIMING_REPETITIONS = 25

HISTOGRAM_X_LABEL = 'Predicted Distance from Ground Truth [m]'
HISTOGRAM_Y_LABEL = 'Normalized frequency of occurence [-]'

PLOT_K_LABEL = 'K Value [-]'
PLOT_FORESTS_LABEL = 'Forests [-]'
PLOT_EXECUTION_TIME_LABEL_S = 'Execution time [s]'
PLOT_EXECUTION_TIME_LABEL_N = 'Execution time [-]'
PLOT_MEAN_ERROR_LABEL_M = 'Mean Error [m]'

SAMPLES_PER_POINT = 50

MAX_K = 251
K = 15

MIN_FORESTS = 20
MAX_FORESTS = 300
FORESTS = 160
FORESTS_STEP = 20

IDX_X = {
    'A':  0,
    'B':  1,
    'C':  2,
    'D':  3,
    'E':  4,
    'F':  5,
    'G':  6,
    'H':  7,
    'I':  8,
    'J':  9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
}

IDX_Y = {
     '1':  0,
     '2':  1,
     '3':  2,
     '4':  3,
     '5':  4,
     '6':  5,
     '7':  6,
     '8':  7,
     '9':  8,
    '10':  9,
    '11': 10,
}

DISTANCE_X = {
    'A':  0.85,
    'B':  1.85,
    'C':  2.85,
    'D':  3.85,
    'E':  4.85,
    'F':  5.85,
    'G':  6.85,
    'H':  7.85,
    'I':  8.85,
    'J':  9.85,
    'K': 10.85,
    'L': 11.85,
    'M': 12.85,
    'N': 13.85,
    'O': 14.85,
    'P': 15.85,
}

DISTANCE_Y = {
     '1':  0.1,
     '2':  1.1,
     '3':  2.1,
     '4':  3.1,
     '5':  4.1,
     '6':  5.1,
     '7':  6.1,
     '8':  7.1,
     '9':  8.1,
    '10':  9.1,
    '11': 10.1,
}

FIELD_NAMES = ['X', 'Y', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4', 'RSSI_5']