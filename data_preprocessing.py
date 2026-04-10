"""
Data Preprocessing Module for Solar Power Prediction.
Contains functions for loading, cleaning, and preparing data for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import (
    YEARS_TO_INCLUDE, 
    FEATURE_COLUMNS, 
    WINDOW_SIZE, 
    TRAIN_SPLIT
)


def load_and_clean_data(filepath):
    """
    Load, clean, and preprocess solar power data for ML model training.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing solar power data.
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed DataFrame ready for model training.
    """
    
    # Load the CSV file with timestamp as index, parsed as dates
    print("Loading data...")
    df = pd.read_csv(filepath, 
                     index_col='timestamp', 
                     parse_dates=True,
                     on_bad_lines='skip')
    print(f"Loaded {len(df):,} rows")
    
    # Filter to only the specified years with good data quality
    print(f"\nFiltering to years {YEARS_TO_INCLUDE}...")
    df = df[df.index.year.isin(YEARS_TO_INCLUDE)]
    print(f"Rows after filtering: {len(df):,}")
    
    # Keep only the required columns
    available_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    df = df[available_columns]
    print(f"Kept columns: {available_columns}")
    
    # Resample to Hourly mean to reduce noise
    print("\nResampling to hourly mean...")
    df = df.resample('h').mean()
    print(f"Rows after resampling: {len(df):,}")
    
    # Clean the data
    print("\nCleaning data...")
    
    # 1. Linear interpolate missing values
    print("  - Interpolating missing values...")
    df = df.interpolate(method='linear')
    
    # 2. Drop any remaining NaNs
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    print(f"  - Dropped {rows_dropped:,} rows with remaining NaNs")
    
    # 3. Clip 'Active_Power' so values < 0 become 0
    if 'Active_Power' in df.columns:
        negative_count = (df['Active_Power'] < 0).sum()
        df['Active_Power'] = df['Active_Power'].clip(lower=0)
        print(f"  - Clipped {negative_count:,} negative Active_Power values to 0")
    
    # 4. Force 'Active_Power' to 0 whenever 'Global_Horizontal_Radiation' is < 1.0 (Night-time logic)
    if 'Active_Power' in df.columns and 'Global_Horizontal_Radiation' in df.columns:
        night_mask = df['Global_Horizontal_Radiation'] < 1.0
        night_count = night_mask.sum()
        df.loc[night_mask, 'Active_Power'] = 0
        print(f"  - Set Active_Power to 0 for {night_count:,} night-time rows (GHR < 1.0)")
    
    # Feature Engineering: Create cyclical time features
    print("\nFeature Engineering...")
    hour = df.index.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
    print("  - Created 'Hour_Sin' and 'Hour_Cos' columns")
    
    # Final summary
    print("\n" + "="*50)
    print("CLEANING COMPLETE")
    print("="*50)
    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    return df


def create_sequences(data, window_size=WINDOW_SIZE):
    """
    Prepare data for a Keras RNN by creating sequences.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Cleaned DataFrame with features. 'Active_Power' should be column 0.
    window_size : int
        Number of past hours to use as input (default from config).
    
    Returns:
    --------
    X_train : np.ndarray
        Training input sequences (past window_size hours of all features).
    X_test : np.ndarray
        Testing input sequences.
    y_train : np.ndarray
        Training targets (Active_Power at the next hour).
    y_test : np.ndarray
        Testing targets.
    scaler : MinMaxScaler
        Fitted scaler object to inverse transform predictions later.
    """
    
    # Scale the data to range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"Window size: {window_size}")
    
    # Create sequences
    X = []
    y = []
    
    # Iterate through the data to create sequences
    # X: past window_size hours of ALL features
    # y: Active_Power (column 0) of the NEXT hour (step window_size + 1)
    for i in range(len(scaled_data) - window_size):
        # Input: rows i to i+window_size (exclusive), all columns
        X.append(scaled_data[i:i + window_size, :])
        # Target: Active_Power (column 0) at the next time step
        y.append(scaled_data[i + window_size, 0])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}")  # (samples, window_size, features)
    print(f"y shape: {y.shape}")  # (samples,)
    
    # Split into Train and Test - NO SHUFFLE for time series
    split_index = int(len(X) * TRAIN_SPLIT)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"\nTrain/Test Split ({int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}, no shuffle):")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Test the module
    from config import RAW_DATA_PATH, CLEANED_DATA_PATH
    
    df = load_and_clean_data(RAW_DATA_PATH)
    print("\nSample data:")
    print(df.head())
    
    # Save cleaned data
    df.to_csv(CLEANED_DATA_PATH)
    print(f"\nSaved to: {CLEANED_DATA_PATH}")
