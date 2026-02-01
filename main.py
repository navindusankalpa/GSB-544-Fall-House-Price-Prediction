from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from pathlib import Path
from typing import Optional
import io
from datetime import datetime
from contextlib import asynccontextmanager

# ============================================================================
# Configuration
# ============================================================================

# Paths to your saved model and preprocessors (cross-platform compatible)
BASE_DIR = Path(__file__).parent.parent
ANN_MODEL_PATH = BASE_DIR / "app" / "model" / "ann" / "0_2_score_model.keras"
ANN_SCALER_PATH = BASE_DIR / "app" / "model" / "ann" / "scaler.joblib"
ANN_LABEL_ENCODERS_PATH = BASE_DIR / "app" / "model" / "ann" / "label_encoders.joblib"
OUTPUT_DIR = BASE_DIR / "app" / "predictions"

XGB_MODEL_PATH = BASE_DIR / "app" / "model" / "xgboost" / "xgboost_model.pkl"
XGB_SCALER_PATH = BASE_DIR / "app" / "model" / "xgboost" / "xgboost_scaler.joblib"
XGB_LABEL_ENCODER_PATH = BASE_DIR / "app" / "model" / "xgboost" / "xgboost_label_encoders.joblib"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using trained neural network model",
    version="1.0.0"
)

# Add CORS middleware to allow requests from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Variables for Model and Preprocessors
# ============================================================================

model = None
scaler = None
label_encoders = None

xgb_model = None
xgb_scaler = None
xgb_label_encoders = None

# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model_and_preprocessors():
    """
    Load the trained model and preprocessors on startup.
    """
    global model, scaler, label_encoders, xgb_model, xgb_scaler, xgb_label_encoders
    
    try:
        # Load Keras model
        print(f"Loading model from {ANN_MODEL_PATH}...")
        model = tf.keras.models.load_model(str(ANN_MODEL_PATH))
        print("Model loaded successfully")
        
        # Load scaler
        print(f"Loading scaler from {ANN_SCALER_PATH}...")
        scaler = joblib.load(str(ANN_SCALER_PATH))
        print("Scaler loaded successfully")
        
        # Load label encoders
        print(f"Loading label encoders from {ANN_LABEL_ENCODERS_PATH}...")
        label_encoders = joblib.load(str(ANN_LABEL_ENCODERS_PATH))
        print("Label encoders loaded successfully")

        print("Loading XGBoost model and components")
        xgb_model = joblib.load(str(XGB_MODEL_PATH))
        xgb_scaler = joblib.load(str(XGB_SCALER_PATH))
        xgb_label_encoders = joblib.load(str(XGB_LABEL_ENCODER_PATH))
        
        print("\n" + "="*60)
        print("All components loaded successfully!")
        print("="*60)
        print(f"Model inputs: {[inp.name for inp in model.inputs]}")
        print(f"Categorical features: {list(label_encoders.keys())}")
        print(f"Numerical features: {list(scaler.feature_names_in_)}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"ERROR loading model components: {e}")
        raise


def preprocess_data(df: pd.DataFrame, model_type='ann') -> dict:
    """
    Preprocess the input dataframe using saved encoders and scaler.
    
    Args:
        df: Input dataframe with raw features
        
    Returns:
        Dictionary with preprocessed inputs ready for model prediction
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Define expected features
    #categorical_features = list(label_encoders.keys())
    categorical_features = list(label_encoders.keys() if model_type == 'ann' else xgb_label_encoders.keys())
    #numerical_features = list(scaler.feature_names_in_)
    numerical_features = list(scaler.feature_names_in_ if model_type == 'ann' else xgb_scaler.feature_names_in_)
    
    # Feature engineering (matching your notebook)
    # Calculate house_age if Year_Built exists and house_age doesn't
    if 'Year_Built' in df.columns and 'house_age' not in df.columns:
        if 'Yr_Sold' in df.columns:
            df['house_age'] = df['Yr_Sold'] - df['Year_Built']
        else:
            # Use current year if Yr_Sold not available
            current_year = datetime.now().year
            df['house_age'] = current_year - df['Year_Built']
    
    # Calculate total_bath if not present
    if 'total_bath' not in df.columns:
        full_bath = df.get('Full_Bath', 0)
        half_bath = df.get('Half_Bath', 0) * 0.5
        df['total_bath'] = full_bath + half_bath
    
    # Encode categorical features
    for feature in categorical_features:
        if feature not in df.columns:
            raise ValueError(f"Missing categorical feature: {feature}")
        
        # Handle unknown categories
        if model_type == 'ann':
            df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                lambda x: label_encoders[feature].transform([x])[0] 
                if x in label_encoders[feature].classes_ 
                else 0  # Unknown category maps to 0
            )

        if model_type == 'xgb':
            df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                lambda x: xgb_label_encoders[feature].transform([x])[0] 
                if x in xgb_label_encoders[feature].classes_ 
                else 0  # Unknown category maps to 0
            )
    
    # Check and prepare numerical features
    missing_numerical = [f for f in numerical_features if f not in df.columns]
    if missing_numerical:
        raise ValueError(f"Missing numerical features: {missing_numerical}")
    
    # Scale numerical features
    if model_type == 'ann':
        df[numerical_features] = scaler.transform(df[numerical_features])
    elif model_type == 'xgb':
        df[numerical_features] = xgb_scaler.transform(df[numerical_features])
    # Prepare inputs for model
    if model_type == 'ann':
        # Keras model needs a dictionary with named inputs
        inputs = {}
        
        # Categorical inputs
        for feature in categorical_features:
            inputs[f'{feature}_input'] = df[f'{feature}_encoded'].values.reshape(-1, 1)
        
        # Numerical input
        inputs['numerical_input'] = df[numerical_features].values.astype('float32')
        
        return inputs
    
    elif model_type == 'xgb':
        # XGBoost needs a DataFrame or 2D array with all features
        # Combine encoded categorical and scaled numerical features
        xgb_inputs = pd.DataFrame()
        
        # Add encoded categorical features
        for feature in categorical_features:
            xgb_inputs[f'{feature}_encoded'] = df[f'{feature}_encoded'].values
        
        # Add scaled numerical features
        for feature in numerical_features:
            xgb_inputs[feature] = df[feature].values
        
        return xgb_inputs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and preprocessors
    load_model_and_preprocessors()
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")

# Then update the FastAPI initialization to:
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using trained neural network model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information (this page)",
            "/health": "Health check",
            "/predict": "Upload CSV file for predictions (POST)",
            "/predict-json": "Send JSON data for predictions (POST)",
            "/model-info": "Get model information"
        },
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API and model are loaded.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_inputs": [inp.name for inp in model.inputs],
        "model_outputs": [out.name for out in model.outputs],
        "categorical_features": list(label_encoders.keys()),
        "numerical_features": list(scaler.feature_names_in_),
        "total_params": model.count_params()
    }


@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Upload a CSV file and get predictions.
    
    The predictions will be saved as a CSV file and returned for download.
    
    Args:
        file: CSV file with house data
        
    Returns:
        FileResponse with predictions CSV
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Clean column names (match your notebook preprocessing)
        df.columns = df.columns.str.replace(" ", "_")
        
        print(f"Received file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # Store PID if exists (for final output)
        has_pid = 'PID' in df.columns
        if has_pid:
            pids = df['PID'].copy()
        else:
            #pids = pd.Series(range(len(df)), name='Id')
            raise HTTPException(status_code=400, detail="File does not contain PID column.")
            return
        
        # Preprocess data
        print("Preprocessing data...")
        inputs = preprocess_data(df, model_type='ann')
        
        # Make predictions
        print("Making predictions...")
        predictions_log = model.predict(inputs).ravel()
        
        # Convert from log scale back to actual prices
        predictions = np.expm1(predictions_log)
        
        print(f"Predictions made: min=${predictions.min():,.2f}, max=${predictions.max():,.2f}")
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'PID' if has_pid else 'Id': pids,
            'SalePrice': predictions
        })
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ann_predictions_{timestamp}.csv"
        output_path = OUTPUT_DIR / output_filename
        output_df.to_csv(str(output_path), index=False)
        
        print(f"Predictions saved to: {output_path}")
        
        # Return file for download
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/xgb_predict")
async def xgb_predict(file: UploadFile = File(...)):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="XGB model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        df.columns = df.columns.str.replace(" ", "_")
        print(f"File: {file.filename} received, columns: {df.columns.to_list()}")

        has_pid = 'PID' in df.columns
        if has_pid:
            pids = df['PID'].copy()
        else:
            #pids = pd.Series(range(len(df)), name='Id')
            raise HTTPException(status_code=400, detail="File does not contain PID column.")
            return
        
        # Preprocess data
        print("Preprocessing data...")
        inputs = preprocess_data(df, model_type='xgb')
        
        # Make predictions
        print("Making predictions...")
        predictions_log = xgb_model.predict(inputs).ravel()
        
        # Convert from log scale back to actual prices
        predictions = np.expm1(predictions_log)
        
        print(f"Predictions made: min=${predictions.min():,.2f}, max=${predictions.max():,.2f}")
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'PID' if has_pid else 'Id': pids,
            'SalePrice': predictions
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"xgb_predictions_{timestamp}.csv"
        output_path = OUTPUT_DIR / output_filename
        output_df.to_csv(str(output_path), index=False)
        
        print(f"Predictions saved to: {output_path}")
        
        # Return file for download
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
    except Exception as e:        
        print(f"Error prediction, {e}")
        raise HTTPException(status_code=500, detail=f"Error prediction, {e}")

@app.get("/predictions")
async def list_predictions():
    """
    List all saved prediction files.
    """
    try:
        files = os.listdir(str(OUTPUT_DIR))
        csv_files = [f for f in files if f.endswith('.csv')]
        
        return {
            "prediction_files": csv_files,
            "count": len(csv_files),
            "directory": str(OUTPUT_DIR)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting House Price Prediction API")
    print("="*60)

    # Run the server
    uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True
)
