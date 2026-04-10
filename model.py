"""
Model Module for Solar Power Prediction.
Contains functions for building, training, evaluating the RNN model,
and analyzing feature importance.
"""

import os
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

# Import all config settings internally - train.py doesn't need to know about these
from config import (
    RNN_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    HOURS_TO_PLOT,
    REPORT_DIR,
    LOSS_CURVE_PATH,
    SCATTER_PLOT_PATH,
    ERROR_DIST_PATH,
    ZOOM_PLOT_PATH,
    METRICS_PATH,
    CORRELATION_HEATMAP_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_SCATTER_PATH,
    FEATURE_ANALYSIS_PATH,
)


def build_and_train_model(X_train, y_train, epochs=None):
    """
    Build and train the SimpleRNN model for solar power prediction.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features with shape (samples, timesteps, features)
    y_train : np.ndarray
        Training target values
    epochs : int, optional
        Number of training epochs (uses config default if not specified)
    
    Returns:
    --------
    tuple : (model, history)
        Trained Keras model and training history
    """
    if epochs is None:
        epochs = EPOCHS
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Building RNN model with input shape: {input_shape}")
    
    # Build the model
    model = Sequential([
        SimpleRNN(RNN_UNITS, activation='tanh', input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation='relu'),
        Dense(1)
    ])
    
    # Compile with Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mse')
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train the model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    print("\nTraining complete!")
    return model, history


def evaluate_and_plot(model, X_test, y_test, scaler, hours_to_plot=None):
    """
    Evaluate the model and create prediction vs actual plot.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target values (scaled)
    scaler : MinMaxScaler
        Scaler used for normalization
    hours_to_plot : int, optional
        Number of hours to show in the plot
    
    Returns:
    --------
    dict : Dictionary containing predictions and metrics
    """
    if hours_to_plot is None:
        hours_to_plot = HOURS_TO_PLOT
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform to get real values
    # Create dummy arrays with correct shape for inverse transform
    n_features = scaler.n_features_in_
    
    # For predictions
    pred_dummy = np.zeros((len(predictions), n_features))
    pred_dummy[:, 0] = predictions.flatten()
    predictions_real = scaler.inverse_transform(pred_dummy)[:, 0]
    
    # For actual values
    actual_dummy = np.zeros((len(y_test), n_features))
    actual_dummy[:, 0] = y_test
    actual_real = scaler.inverse_transform(actual_dummy)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(actual_real, predictions_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_real, predictions_real)
    r2 = r2_score(actual_real, predictions_real)
    
    print(f"\nModel Performance (Real Scale):")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 6))
    plt.plot(actual_real[:hours_to_plot], label='Actual', alpha=0.8)
    plt.plot(predictions_real[:hours_to_plot], label='Predicted', alpha=0.8)
    plt.title(f'Solar Power Prediction vs Actual (First {hours_to_plot} Hours)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Active Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'prediction_plot.png'), dpi=300)
    plt.close()
    
    return {
        'predictions': predictions_real,
        'actual': actual_real,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_training_history(history):
    """
    Plot training and validation loss curves.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history from model.fit()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()
    print(f"[OK] Training history plot saved")


def generate_report(model, history, X_test, y_test, scaler):
    """
    Generate comprehensive report with visualizations for presentation.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    history : keras.callbacks.History
        Training history
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target values (scaled)
    scaler : MinMaxScaler
        Scaler used for normalization
    
    Returns:
    --------
    dict : Dictionary containing all results and file paths
    """
    # Create reports directory if it doesn't exist
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Saving reports to: {REPORT_DIR}")
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform to get real values
    n_features = scaler.n_features_in_
    
    pred_dummy = np.zeros((len(predictions), n_features))
    pred_dummy[:, 0] = predictions.flatten()
    predictions_real = scaler.inverse_transform(pred_dummy)[:, 0]
    
    actual_dummy = np.zeros((len(y_test), n_features))
    actual_dummy[:, 0] = y_test
    actual_real = scaler.inverse_transform(actual_dummy)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(actual_real, predictions_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_real, predictions_real)
    r2 = r2_score(actual_real, predictions_real)
    
    # Save metrics to text file
    with open(METRICS_PATH, 'w') as f:
        f.write("="*50 + "\n")
        f.write("SOLAR POWER PREDICTION - MODEL METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"R-squared (R2):           {r2:.4f} ({r2*100:.2f}%)\n")
        f.write(f"Mean Absolute Error:      {mae:.4f} kW\n")
        f.write(f"Root Mean Squared Error:  {rmse:.4f} kW\n")
        f.write(f"Mean Squared Error:       {mse:.4f}\n")
        f.write(f"\nTest Samples: {len(y_test):,}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
    print(f"[OK] Metrics saved to: {METRICS_PATH}")
    
    # =========================================================================
    # VISUALIZATION 1: Training Loss Curve
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    plt.plot(epochs_range, history.history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs_range, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    plt.title('Model Loss During Training', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    min_val_loss = min(history.history['val_loss'])
    min_epoch = history.history['val_loss'].index(min_val_loss) + 1
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                 xy=(min_epoch, min_val_loss),
                 xytext=(min_epoch + 2, min_val_loss + 0.001),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Loss curve saved to: {LOSS_CURVE_PATH}")
    
    # =========================================================================
    # VISUALIZATION 2: Scatter Plot (Actual vs Predicted)
    # =========================================================================
    plt.figure(figsize=(10, 10))
    
    plt.scatter(actual_real, predictions_real, alpha=0.5, s=10, c='steelblue', label='Predictions')
    
    max_val = max(max(actual_real), max(predictions_real))
    min_val = min(min(actual_real), min(predictions_real))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    plt.title('Actual vs Predicted Power (Regression Plot)', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Power (kW)', fontsize=12)
    plt.ylabel('Predicted Power (kW)', fontsize=12)
    plt.legend(fontsize=11)
    
    plt.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Scatter plot saved to: {SCATTER_PLOT_PATH}")
    
    # =========================================================================
    # VISUALIZATION 3: Error Distribution
    # =========================================================================
    errors = predictions_real - actual_real
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Error (Predicted - Actual) kW', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_title('Error Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Error (kW)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ERROR_DIST_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Error distribution saved to: {ERROR_DIST_PATH}")
    
    # =========================================================================
    # VISUALIZATION 4: Zoomed Time Series
    # =========================================================================
    hours = HOURS_TO_PLOT
    
    plt.figure(figsize=(14, 6))
    
    time_range = range(hours)
    plt.plot(time_range, actual_real[:hours], 'b-', linewidth=1.5, label='Actual Power', alpha=0.8)
    plt.plot(time_range, predictions_real[:hours], 'r-', linewidth=1.5, label='Predicted Power', alpha=0.8)
    
    plt.fill_between(time_range, actual_real[:hours], predictions_real[:hours], 
                     alpha=0.2, color='gray', label='Prediction Gap')
    
    plt.title(f'Solar Power: Actual vs Predicted (First {hours} Hours)', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Active Power (kW)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ZOOM_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Zoom plot saved to: {ZOOM_PLOT_PATH}")
    
    print(f"\n[OK] All visualizations saved to: {REPORT_DIR}")
    
    return {
        'predictions': predictions_real,
        'actual': actual_real,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'report_dir': REPORT_DIR
    }


def analyze_feature_importance(model, X_test, y_test, scaler, df):
    """
    Analyze which weather features most affect solar power production.
    Creates visualizations and a detailed report for presentation.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : np.ndarray
        Test features with shape (samples, timesteps, features)
    y_test : np.ndarray
        Test target values (scaled)
    scaler : MinMaxScaler
        Scaler used for normalization
    df : pd.DataFrame
        Original cleaned dataframe with column names
    
    Returns:
    --------
    dict : Dictionary containing feature importance analysis results
    """
    print("\nAnalyzing feature importance...")
    
    # Get feature names from the dataframe
    feature_names = df.columns.tolist()
    print(f"Features: {feature_names}")
    
    # =========================================================================
    # 1. CORRELATION ANALYSIS
    # =========================================================================
    print("\n1. Computing correlations with Active_Power...")
    
    correlations = df.corr()['Active_Power'].drop('Active_Power').sort_values(ascending=False)
    
    print("\nCorrelation with Active_Power:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CORRELATION_HEATMAP_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Correlation heatmap saved to: {CORRELATION_HEATMAP_PATH}")
    
    # =========================================================================
    # 2. PERMUTATION IMPORTANCE
    # =========================================================================
    print("\n2. Computing permutation importance...")
    
    # Get baseline MSE
    baseline_pred = model.predict(X_test, verbose=0)
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    
    importance_scores = {}
    
    for i, feature in enumerate(feature_names):
        # Create a copy of X_test
        X_permuted = X_test.copy()
        
        # Shuffle the feature across all timesteps
        np.random.seed(42)
        for t in range(X_test.shape[1]):
            np.random.shuffle(X_permuted[:, t, i])
        
        # Predict with permuted feature
        permuted_pred = model.predict(X_permuted, verbose=0)
        permuted_mse = mean_squared_error(y_test, permuted_pred)
        
        # Importance = increase in MSE when feature is shuffled
        importance = permuted_mse - baseline_mse
        importance_scores[feature] = importance
        print(f"  {feature}: MSE increase = {importance:.6f}")
    
    # Sort by importance
    sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    # Create importance bar chart
    plt.figure(figsize=(12, 6))
    
    features = list(sorted_importance.keys())
    scores = list(sorted_importance.values())
    colors = ['#e74c3c' if s > 0 else '#3498db' for s in scores]
    
    bars = plt.barh(features, scores, color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Importance (MSE Increase When Shuffled)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance for Solar Power Prediction\n(Higher = More Important)', 
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        plt.text(score + 0.0001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Feature importance plot saved to: {FEATURE_IMPORTANCE_PATH}")
    
    # =========================================================================
    # 3. SCATTER PLOTS: Each Feature vs Active Power
    # =========================================================================
    print("\n3. Creating feature vs power scatter plots...")
    
    # Get features excluding target and cyclical time features
    plot_features = [f for f in feature_names if f not in ['Active_Power', 'Hour_Sin', 'Hour_Cos']]
    
    n_features = len(plot_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
    
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, plot_features):
        ax.scatter(df[feature], df['Active_Power'], alpha=0.3, s=5, c='steelblue')
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Active Power (kW)', fontsize=11)
        
        # Add correlation value
        corr = df[feature].corr(df['Active_Power'])
        ax.set_title(f'{feature}\n(r = {corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weather Features vs Solar Power Production', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FEATURE_SCATTER_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Feature scatter plots saved to: {FEATURE_SCATTER_PATH}")
    
    # =========================================================================
    # 4. GENERATE TEXT REPORT
    # =========================================================================
    top_feature = list(sorted_importance.keys())[0]
    top_importance = list(sorted_importance.values())[0]
    
    with open(FEATURE_ANALYSIS_PATH, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("Solar Power Prediction Model\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. CORRELATION WITH ACTIVE POWER\n")
        f.write("-"*40 + "\n")
        for feature, corr in correlations.items():
            bar = "+" * int(abs(corr) * 20)
            sign = "+" if corr > 0 else "-"
            f.write(f"  {feature:35s}: {corr:+.4f} {bar}\n")
        
        f.write("\n2. PERMUTATION IMPORTANCE\n")
        f.write("-"*40 + "\n")
        f.write("(Higher value = feature is more important for predictions)\n\n")
        for i, (feature, importance) in enumerate(sorted_importance.items(), 1):
            f.write(f"  {i}. {feature:35s}: {importance:.6f}\n")
        
        f.write(f"\n3. KEY FINDINGS\n")
        f.write("-"*40 + "\n")
        f.write(f"  - Most important feature: {top_feature}\n")
        f.write(f"  - Its permutation importance: {top_importance:.6f}\n")
        
        # Find highest correlated feature
        highest_corr_feature = correlations.idxmax()
        highest_corr = correlations.max()
        f.write(f"  - Highest correlation: {highest_corr_feature} (r={highest_corr:.4f})\n")
        
        f.write("\n4. INTERPRETATION\n")
        f.write("-"*40 + "\n")
        if 'Global_Horizontal_Radiation' in sorted_importance:
            if list(sorted_importance.keys())[0] == 'Global_Horizontal_Radiation':
                f.write("  Solar radiation (GHR) is the dominant factor affecting power\n")
                f.write("  production, which aligns with solar panel physics.\n")
        
        f.write("\n5. GENERATED FILES\n")
        f.write("-"*40 + "\n")
        f.write(f"  - Correlation Heatmap: {CORRELATION_HEATMAP_PATH}\n")
        f.write(f"  - Feature Importance: {FEATURE_IMPORTANCE_PATH}\n")
        f.write(f"  - Scatter Plots: {FEATURE_SCATTER_PATH}\n")
        f.write(f"  - This Report: {FEATURE_ANALYSIS_PATH}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"[OK] Analysis report saved to: {FEATURE_ANALYSIS_PATH}")
    
    # Print summary
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*50)
    print(f"Most Important Feature: {top_feature}")
    print(f"Highest Correlation: {highest_corr_feature} (r={highest_corr:.4f})")
    
    return {
        'correlations': correlations.to_dict(),
        'permutation_importance': sorted_importance,
        'top_important': top_feature,
        'highest_correlated': highest_corr_feature,
        'report_path': FEATURE_ANALYSIS_PATH
    }
