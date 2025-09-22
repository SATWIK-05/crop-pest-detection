import pandas as pd

# IMPORTANT: Make sure this filename matches what you have.
# It might be 'CropDataset.csv' or 'Crop_recommendation.csv'
file_name = 'Crop_recommendation.csv'

try:
    # We will try to read the file and print what it sees
    data = pd.read_csv(file_name, encoding='latin1')

    print("--- File read successfully. Here are the first 5 rows: ---")
    print(data.head())

    print("\n--- Here are the column names found: ---")
    print(data.columns)

except Exception as e:
    print(f"Could not read the file. Error: {e}")