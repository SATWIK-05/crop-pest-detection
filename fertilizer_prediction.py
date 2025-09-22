import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

def train_model(file_path):
    """
    Trains a RandomForestClassifier model with improved preprocessing and balancing.
    """
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
        print("Available files in directory:")
        import os
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for file in files:
            print(f"  - {file}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # CRITICAL FIX: Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Display dataset info
    print("\nDataset Info:")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Automatically identify features and the target variable
    features = df.columns[:-1].tolist()
    target = df.columns[-1]
    
    print(f"\nFeatures: {features}")
    print(f"Target: {target}")
    
    X = df[features]
    y = df[target]

    # Check for class imbalance
    target_distribution = y.value_counts()
    print(f"\nTarget distribution:\n{target_distribution}")
    
    # Preprocess categorical features using LabelEncoder
    label_encoders = {}
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"{col} classes: {list(le.classes_)}")

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"Target classes: {list(le_target.classes_)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale numerical features (excluding categorical ones)
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    scaler = StandardScaler()
    if numerical_features:
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Train model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

    return model, label_encoders, le_target, features, scaler

def get_user_input(features, label_encoders):
    """Get and validate user input with better error handling"""
    user_input = {}
    
    print("\n" + "="*60)
    print("FERTILIZER RECOMMENDATION SYSTEM")
    print("="*60)
    
    for feature in features:
        while True:
            try:
                if feature in label_encoders:
                    # For categorical features
                    encoder = label_encoders[feature]
                    valid_options = list(encoder.classes_)
                    print(f"\nValid options for {feature}: {valid_options}")
                    
                    value = input(f"Enter {feature}: ").strip().title()
                    
                    if value not in valid_options:
                        print(f"Warning: '{value}' not in predefined options. Using anyway...")
                        # Map to closest option or use default
                        if value == "":
                            value = valid_options[0]  # Use first option if empty
                    
                    user_input[feature] = value
                else:
                    # For numerical features
                    value = float(input(f"Enter {feature}: "))
                    user_input[feature] = value
                
                break
                
            except ValueError:
                print("Invalid input! Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None
    
    return user_input

def main():
    # Try both possible filenames
    possible_filenames = [
        "fertilizer_pediction_dataset.csv",  # Original with typo
        "fertilizer_prediction_dataset.csv",  # Correct spelling
        "fertilizer_dataset.csv",
        "dataset.csv"
    ]
    
    file_path = None
    for filename in possible_filenames:
        try:
            pd.read_csv(filename)
            file_path = filename
            print(f"Found dataset: {filename}")
            break
        except:
            continue
    
    if file_path is None:
        print("Could not find the dataset file. Please make sure one of these files exists:")
        for filename in possible_filenames:
            print(f"  - {filename}")
        sys.exit(1)
    
    print("Training the model with improved preprocessing...")
    model, encoders, le_target, features, scaler = train_model(file_path)
    print("\nModel training complete! Ready to make a recommendation.")
    
    # Print valid categorical values for the user
    for col, encoder in encoders.items():
        print(f"Valid {col}: {list(encoder.classes_)}")
    
    # Main prediction loop
    while True:
        try:
            user_input = get_user_input(features, encoders)
            
            if user_input is None:
                break
            
            # Prepare input data
            input_data = []
            for feature in features:
                if feature in encoders:
                    # Encode categorical feature
                    encoder = encoders[feature]
                    try:
                        encoded_value = encoder.transform([user_input[feature]])[0]
                    except ValueError:
                        # If value not seen during training, use most common value
                        encoded_value = 0
                    input_data.append(encoded_value)
                else:
                    input_data.append(user_input[feature])
            
            # Create DataFrame
            new_data = pd.DataFrame([input_data], columns=features)
            
            # Scale numerical features
            numerical_features = new_data.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_features:
                new_data[numerical_features] = scaler.transform(new_data[numerical_features])
            
            # Predict the fertilizer type
            predicted_fertilizer_encoded = model.predict(new_data)
            predicted_fertilizer = le_target.inverse_transform(predicted_fertilizer_encoded)
            
            # Get prediction probabilities
            probabilities = model.predict_proba(new_data)[0]
            confidence = max(probabilities)
            
            # Get top 3 predictions
            top_3_indices = probabilities.argsort()[-3:][::-1]
            top_3_fertilizers = le_target.inverse_transform(top_3_indices)
            top_3_probabilities = probabilities[top_3_indices]
            
            print("\n" + "="*60)
            print("RECOMMENDATION RESULTS")
            print("="*60)
            print(f"Primary Recommendation: {predicted_fertilizer[0]} (Confidence: {confidence:.1%})")
            
            if len(top_3_fertilizers) > 1:
                print(f"\nAlternative Recommendations:")
                for i, (fert, prob) in enumerate(zip(top_3_fertilizers[1:], top_3_probabilities[1:]), 2):
                    print(f"{i}. {fert} (Confidence: {prob:.1%})")
            
            print("="*60)
            
            # Ask for another prediction
            while True:
                another_prediction = input("\nMake another prediction? (yes/no): ").lower().strip()
                if another_prediction in ['yes', 'y', 'no', 'n']:
                    break
                print("Please enter 'yes' or 'no'")
            
            if another_prediction in ['no', 'n']:
                print("Thank you for using the Fertilizer Recommendation System!")
                break

        except ValueError:
            print("Invalid input. Please ensure numeric values are entered correctly.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()35