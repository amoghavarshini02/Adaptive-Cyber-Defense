import pandas as pd
import joblib

# Load processed dataset and trained model's feature list
data = pd.read_csv("data/processed/MedSec-25-cleaned.csv")
feature_columns = joblib.load("models/feature_columns.pkl")

# Select 10 real records with exactly the model's required features
sample = data[feature_columns].head(10)

# Save the demo input file
sample.to_csv("data/processed/demo_sample.csv", index=False)

print("Demo sample created successfully!")
print("Records:", len(sample))
print("Features:", len(sample.columns))
print("Saved to: data/processed/demo_sample.csv")