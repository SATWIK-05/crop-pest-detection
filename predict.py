import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os

# Ignore potential warnings
warnings.filterwarnings("ignore")

def load_data():
    """Load and validate the dataset"""
    file_path = 'Crop_recommendation.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        print("Please ensure the file is in the same directory as this script.")
        return None, None
    
    try:
        data = pd.read_csv(file_path, encoding='latin1')
        print(f"Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data, True
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

def prepare_features(data):
    """Prepare feature columns"""
    expected_features = [
        'Latitude', 'Longitude', 
        'Nitrogen - High', 'Nitrogen - Medium', 'Nitrogen - Low',
        'Phosphorous - High', 'Phosphorous - Medium', 'Phosphorous - Low',
        'Potassium - High', 'Potassium - Medium', 'Potassium - Low',
        'pH - Acidic', 'pH - Neutral', 'pH - Alkaline'
    ]
    
    # Use only features that exist in the dataset
    features = [col for col in expected_features if col in data.columns]
    
    if not features:
        print("Error: No required feature columns found.")
        return None, None
    
    missing_features = set(expected_features) - set(features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    target = 'Crop'
    if target not in data.columns:
        print(f"Error: Target column '{target}' not found.")
        return None, None
    
    X = data[features].copy()
    y = data[target]
    
    return X, y

def clean_data(X):
    """Clean the feature data"""
    print("Cleaning data...")
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(str).str.replace('%', '', regex=False)
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert column {col}: {e}")
    
    # Drop rows with missing values
    initial_rows = X.shape[0]
    X = X.dropna()
    final_rows = X.shape[0]
    
    if initial_rows != final_rows:
        print(f"Removed {initial_rows - final_rows} rows with missing values.")
    
    print("Data cleaning complete.")
    return X

def get_user_input():
    """Get and validate user input"""
    print("\n" + "="*50)
    print("Enter Soil and Location Details for Crop Prediction")
    print("="*50)
    
    try:
        # Get location data
        lat = float(input("Enter Latitude (e.g., 19.07): "))
        lon = float(input("Enter Longitude (e.g., 72.87): "))
        
        print("\nNote: For nutrients and pH, enter percentages (0-100).")
        
        # Get nutrient data
        n_h = float(input("Enter Nitrogen - High level (%): "))
        n_m = float(input("Enter Nitrogen - Medium level (%): "))
        n_l = float(input("Enter Nitrogen - Low level (%): "))

        p_h = float(input("Enter Phosphorous - High level (%): "))
        p_m = float(input("Enter Phosphorous - Medium level (%): "))
        p_l = float(input("Enter Phosphorous - Low level (%): "))

        k_h = float(input("Enter Potassium - High level (%): "))
        k_m = float(input("Enter Potassium - Medium level (%): "))
        k_l = float(input("Enter Potassium - Low level (%): "))

        ph_a = float(input("Enter pH - Acidic level (%): "))
        ph_n = float(input("Enter pH - Neutral level (%): "))
        ph_k = float(input("Enter pH - Alkaline level (%): "))
        
        # Validate percentages
        nutrient_sums = [
            (n_h + n_m + n_l, "Nitrogen"),
            (p_h + p_m + p_l, "Phosphorous"), 
            (k_h + k_m + k_l, "Potassium"),
            (ph_a + ph_n + ph_k, "pH")
        ]
        
        for total, nutrient in nutrient_sums:
            if not (95 <= total <= 105):  # Allow small rounding errors
                print(f"Warning: {nutrient} percentages sum to {total}%, should be close to 100%")
        
        return [[lat, lon, n_h, n_m, n_l, p_h, p_m, p_l, k_h, k_m, k_l, ph_a, ph_n, ph_k]]
        
    except ValueError:
        print("\nInvalid input. Please enter numbers only.")
        return None
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return None

def main():
    """Main function"""
    # Load data
    data, success = load_data()
    if not success:
        return
    
    # Prepare features
    X, y = prepare_features(data)
    if X is None:
        return
    
    # Clean data
    X = clean_data(X)
    
    # Ensure we have matching target values
    y = y.iloc[X.index]
    
    # Split and train model
    print("\nTraining model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    # Get user input and predict
    user_data = get_user_input()
    if user_data is None:
        return
    
    try:
        prediction = model.predict(user_data)
        probabilities = model.predict_proba(user_data)
        top_3_indices = probabilities[0].argsort()[-3:][::-1]
        top_3_crops = model.classes_[top_3_indices]
        top_3_probs = probabilities[0][top_3_indices]
        
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Recommended crop: {prediction[0]}")
        print(f"\nTop 3 recommendations:")
        for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs), 1):
            print(f"{i}. {crop} ({prob*100:.1f}% confidence)")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()