"""
Configuration file for Solar Power Prediction ML Project.
Contains all constants, hyperparameters, and file paths.
"""

import os

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input data
RAW_DATA_PATH = os.path.join(BASE_DIR, '59-Site_DKA-M19_C-Phase.csv')

# Output files
CLEANED_DATA_PATH = os.path.join(BASE_DIR, '59-Site_cleaned_for_ml.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'solar_rnn_model.keras')
PLOT_PATH = os.path.join(BASE_DIR, 'prediction_plot.png')

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# Years to include (using years with good data quality)
YEARS_TO_INCLUDE = [2022, 2023]

# Columns to keep for model training
FEATURE_COLUMNS = [
    'Active_Power', 
    'Global_Horizontal_Radiation', 
    'Weather_Temperature_Celsius', 
    'Weather_Relative_Humidity'
]

# Target column (must be first in FEATURE_COLUMNS)
TARGET_COLUMN = 'Active_Power'

# =============================================================================
# SEQUENCE PARAMETERS
# =============================================================================
WINDOW_SIZE = 24  # Hours of history to use as input
TRAIN_SPLIT = 0.8  # 80% training, 20% testing

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
RNN_UNITS = 64
DROPOUT_RATE = 0.2
DENSE_UNITS = 32
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# =============================================================================
# VISUALIZATION
# =============================================================================
HOURS_TO_PLOT = 150

# =============================================================================
# REPORT OUTPUT PATHS
# =============================================================================
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
LOSS_CURVE_PATH = os.path.join(REPORT_DIR, 'loss_curve.png')
SCATTER_PLOT_PATH = os.path.join(REPORT_DIR, 'scatter_plot.png')
ERROR_DIST_PATH = os.path.join(REPORT_DIR, 'error_distribution.png')
ZOOM_PLOT_PATH = os.path.join(REPORT_DIR, 'zoom_plot.png')
METRICS_PATH = os.path.join(REPORT_DIR, 'metrics.txt')

# Feature Analysis
CORRELATION_HEATMAP_PATH = os.path.join(REPORT_DIR, 'correlation_heatmap.png')
FEATURE_IMPORTANCE_PATH = os.path.join(REPORT_DIR, 'feature_importance.png')
FEATURE_SCATTER_PATH = os.path.join(REPORT_DIR, 'feature_vs_power.png')
FEATURE_ANALYSIS_PATH = os.path.join(REPORT_DIR, 'feature_analysis.txt')
