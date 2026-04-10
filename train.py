"""
Main Training Script for Solar Power Prediction.
Orchestrates data loading, preprocessing, model training, and evaluation.
"""

import numpy as np

from config import RAW_DATA_PATH, CLEANED_DATA_PATH, WINDOW_SIZE, EPOCHS, MODEL_PATH
from data_preprocessing import load_and_clean_data, create_sequences
from model import build_and_train_model, evaluate_and_plot, plot_training_history, generate_report, analyze_feature_importance


def main():
    """Main function to run the complete ML pipeline."""
    
    print("="*60)
    print("SOLAR POWER PREDICTION - RNN MODEL")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Load and Clean Data
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("="*60)
    
    df = load_and_clean_data(RAW_DATA_PATH)
    
    # Display sample
    print("\nSample data (first 5 rows):")
    print(df.head().to_string())
    
    # Save cleaned data
    df.to_csv(CLEANED_DATA_PATH)
    print(f"\nSaved cleaned data to: {CLEANED_DATA_PATH}")
    
    # =========================================================================
    # STEP 2: Create Sequences for RNN
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: CREATING SEQUENCES FOR RNN")
    print("="*60)
    
    X_train, X_test, y_train, y_test, scaler = create_sequences(df, window_size=WINDOW_SIZE)
    
    print(f"\nReady for training!")
    print(f"Input shape: {X_train.shape[1:]} (window_size, features)")
    
    # =========================================================================
    # STEP 3: Build and Train Model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: BUILDING AND TRAINING MODEL")
    print("="*60)
    
    model, history = build_and_train_model(X_train, y_train, epochs=EPOCHS)
    
    # Save the trained model
    model.save(MODEL_PATH)
    print(f"\n[OK] Model saved to: {MODEL_PATH}")
    
    # =========================================================================
    # STEP 4: Evaluate Model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: EVALUATING MODEL")
    print("="*60)
    
    # Basic evaluation
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE Loss: {test_loss:.6f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.6f}")
    
    # =========================================================================
    # STEP 5: Generate Report (Visualizations for Presentation)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: GENERATING REPORT FOR PRESENTATION")
    print("="*60)
    
    results = generate_report(model, history, X_test, y_test, scaler)
    
    # =========================================================================
    # STEP 6: Feature Importance Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: ANALYZING FEATURE IMPORTANCE")
    print("="*60)
    
    feature_analysis = analyze_feature_importance(model, X_test, y_test, scaler, df)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("[SUCCESS] TRAINING COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"Data: {len(df):,} hourly samples (2022-2023)")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"\nModel Performance:")
    print(f"  R² Score: {results['r2_score']:.4f} ({results['r2_score']*100:.2f}%)")
    print(f"  MAE:      {results['mae']:.4f} kW")
    print(f"  RMSE:     {results['rmse']:.4f} kW")
    print(f"\nTop Feature: {feature_analysis['top_important']}")
    print(f"Report files saved to: {results['report_dir']}")
    print("="*60)
    
    return model, history, results, scaler, feature_analysis


if __name__ == "__main__":
    model, history, results, scaler, feature_analysis = main()
