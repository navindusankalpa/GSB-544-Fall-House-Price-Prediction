"""
Test Client for House Price Prediction API
===========================================

This script demonstrates how to interact with the FastAPI service.

Usage:
    python test_client.py
"""

import requests
import pandas as pd
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_info():
    """Get model information."""
    print("\n" + "="*60)
    print("Getting Model Information")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_from_file(file_path: str):
    """
    Test prediction from CSV file.
    
    Args:
        file_path: Path to CSV file containing house data
    """
    print("\n" + "="*60)
    print(f"Testing Prediction from File: {file_path}")
    print("="*60)
    
    try:
        # Open and send file
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Save the returned CSV
            output_file = "downloaded_predictions.csv"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Predictions saved to: {output_file}")
            
            # Display first few predictions
            df = pd.read_csv(output_file)
            print(f"\nFirst 5 predictions:")
            print(df.head())
            print(f"\nStatistics:")
            print(f"  Total predictions: {len(df)}")
            print(f"  Min price: ${df['SalePrice'].min():,.2f}")
            print(f"  Max price: ${df['SalePrice'].max():,.2f}")
            print(f"  Mean price: ${df['SalePrice'].mean():,.2f}")
            
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_predict_from_json():
    """Test prediction from JSON data."""
    print("\n" + "="*60)
    print("Testing Prediction from JSON")
    print("="*60)
    
    # Sample data (adjust values based on your features)
    sample_data = {
        "data": [
            {
                "Neighborhood": "NAmes",
                "Overall_Qual": 7,
                "Gr_Liv_Area": 1710,
                "Lot_Area": 9550,
                "house_age": 20
            },
            {
                "Neighborhood": "Edwards",
                "Overall_Qual": 5,
                "Gr_Liv_Area": 1200,
                "Lot_Area": 7500,
                "house_age": 35
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict-json", json=sample_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  House {i+1}: ${pred:,.2f}")
        print(f"\nStatistics: {json.dumps(result['statistics'], indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_list_predictions():
    """List all saved prediction files."""
    print("\n" + "="*60)
    print("Listing Saved Predictions")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/predictions")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total files: {result['count']}")
        print(f"Files: {result['prediction_files']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def create_sample_test_file():
    """
    Create a sample CSV file for testing.
    Returns the filename.
    """
    print("\n" + "="*60)
    print("Creating Sample Test File")
    print("="*60)
    
    # Create sample data
    sample_data = {
        'PID': [123456789, 987654321],
        'Neighborhood': ['NAmes', 'Edwards'],
        'Overall_Qual': [7, 5],
        'Gr_Liv_Area': [1710, 1200],
        'Lot_Area': [9550, 7500],
        'house_age': [20, 35]
    }
    
    df = pd.DataFrame(sample_data)
    filename = "sample_test_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"✓ Created sample file: {filename}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    return filename


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HOUSE PRICE PREDICTION API - TEST CLIENT")
    print("="*60)
    print(f"API URL: {BASE_URL}")
    print("Make sure the API server is running!")
    print("="*60)
    
    try:
        # Test 1: Health check
        if not test_health_check():
            print("\n❌ Health check failed. Is the server running?")
            return
        
        # Test 2: Model info
        if not test_model_info():
            print("\n❌ Failed to get model info")
            return
        
        # Test 3: Create sample file
        sample_file = create_sample_test_file()
        
        # Test 4: Predict from file
        test_predict_from_file(sample_file)
        
        # Test 5: Predict from JSON
        test_predict_from_json()
        
        # Test 6: List predictions
        test_list_predictions()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running with: python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
