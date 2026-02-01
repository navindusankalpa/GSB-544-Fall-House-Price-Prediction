"""
Standalone Batch Prediction Script
===================================

This script loads your model and makes predictions on a CSV file
WITHOUT needing to run the FastAPI server.

Use this for:
- Quick one-off predictions
- Batch processing multiple files
- Automated pipelines

Usage:
    python batch_predict.py input_file.csv output_file.csv
    python batch_predict.py input_file.csv  # Auto-generates output filename
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import sys
from pathlib import Path
from datetime import datetime
import argparse


class HousePricePredictor:
    """
    House price prediction model wrapper.
    """
    
    def __init__(self, model_path, scaler_path, encoders_path):
        """
        Initialize the predictor with saved model and preprocessors.
        
        Args:
            model_path: Path to saved Keras model
            scaler_path: Path to saved scaler
            encoders_path: Path to saved label encoders
        """
        print("Loading model components...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
        
        # Load encoders
        self.label_encoders = joblib.load(encoders_path)
        print(f"✓ Label encoders loaded from {encoders_path}")
        
        # Extract feature information
        self.categorical_features = list(self.label_encoders.keys())
        self.numerical_features = list(self.scaler.feature_names_in_)
        
        print(f"\nModel Configuration:")
        print(f"  Categorical features: {self.categorical_features}")
        print(f"  Numerical features: {self.numerical_features}")
        print()
    
    def preprocess(self, df):
        """
        Preprocess the input dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with preprocessed inputs
        """
        df = df.copy()
        
        # Feature engineering
        # Calculate house_age if needed
        if 'house_age' not in df.columns:
            if 'Year_Built' in df.columns:
                if 'Yr_Sold' in df.columns:
                    df['house_age'] = df['Yr_Sold'] - df['Year_Built']
                else:
                    current_year = datetime.now().year
                    df['house_age'] = current_year - df['Year_Built']
        
        # Calculate total_bath if needed
        if 'total_bath' not in df.columns:
            if 'Full_Bath' in df.columns or 'Half_Bath' in df.columns:
                full_bath = df.get('Full_Bath', 0)
                half_bath = df.get('Half_Bath', 0) * 0.5
                df['total_bath'] = full_bath + half_bath
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required categorical feature: {feature}")
            
            df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                lambda x: self.label_encoders[feature].transform([x])[0]
                if x in self.label_encoders[feature].classes_
                else 0
            )
        
        # Check numerical features
        missing = [f for f in self.numerical_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required numerical features: {missing}")
        
        # Scale numerical features
        df[self.numerical_features] = self.scaler.transform(df[self.numerical_features])
        
        # Prepare inputs
        inputs = {}
        
        for feature in self.categorical_features:
            inputs[f'{feature}_input'] = df[f'{feature}_encoded'].values.reshape(-1, 1)
        
        inputs['numerical_input'] = df[self.numerical_features].values.astype('float32')
        
        return inputs
    
    def predict(self, df):
        """
        Make predictions on the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Array of predictions
        """
        inputs = self.preprocess(df)
        predictions_log = self.model.predict(inputs, verbose=0).ravel()
        predictions = np.expm1(predictions_log)
        return predictions
    
    def predict_from_file(self, input_path, output_path=None):
        """
        Make predictions from CSV file and save to output file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file (optional)
            
        Returns:
            Path to output file
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")
        
        # Read input file
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df)} rows")
        
        # Clean column names
        df.columns = df.columns.str.replace(" ", "_")
        
        # Store ID column if exists
        has_pid = 'PID' in df.columns
        if has_pid:
            ids = df['PID'].copy()
        else:
            ids = pd.Series(range(len(df)), name='Id')
        
        # Make predictions
        print("Making predictions...")
        predictions = self.predict(df)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'PID' if has_pid else 'Id': ids,
            'SalePrice': predictions
        })
        
        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(input_path).stem
            output_path = f"{input_name}_predictions_{timestamp}.csv"
        
        # Save predictions
        output_df.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved to: {output_path}")
        print(f"\nPrediction Statistics:")
        print(f"  Count: {len(predictions)}")
        print(f"  Min:   ${predictions.min():,.2f}")
        print(f"  Max:   ${predictions.max():,.2f}")
        print(f"  Mean:  ${predictions.mean():,.2f}")
        print(f"  Median: ${np.median(predictions):,.2f}")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    """Main function with command line interface."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Make house price predictions from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_predict.py input.csv
  python batch_predict.py input.csv output.csv
  python batch_predict.py data/test.csv predictions/results.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input CSV file with house data'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        default=None,
        help='Path to output CSV file (optional, auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--model',
        default='0_2_score_model.keras',
        help='Path to model file (default: 0_2_score_model.keras)'
    )
    
    parser.add_argument(
        '--scaler',
        default='scaler.joblib',
        help='Path to scaler file (default: scaler.joblib)'
    )
    
    parser.add_argument(
        '--encoders',
        default='label_encoders.joblib',
        help='Path to encoders file (default: label_encoders.joblib)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("HOUSE PRICE PREDICTION - BATCH PROCESSING")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = HousePricePredictor(
            model_path=args.model,
            scaler_path=args.scaler,
            encoders_path=args.encoders
        )
        
        # Make predictions
        output_path = predictor.predict_from_file(
            input_path=args.input_file,
            output_path=args.output_file
        )
        
        print("✓ SUCCESS!")
        print(f"Results saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR: Invalid data - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("HOUSE PRICE PREDICTION - BATCH PROCESSING")
        print("="*60)
        print("\nUsage:")
        print("  python batch_predict.py <input_file.csv> [output_file.csv]")
        print("\nExamples:")
        print("  python batch_predict.py test_data.csv")
        print("  python batch_predict.py test_data.csv predictions.csv")
        print("\nFor more options, use: python batch_predict.py --help")
        print("="*60 + "\n")
        sys.exit(0)
    
    main()
