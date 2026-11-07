
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

required_files = [
    "data/raw/MedSec-25.csv",
    "data/processed/MedSec-25-cleaned.csv",
    "data/processed/demo_sample.csv",
    "models/cyber_model.pkl",
    "models/scaler.pkl",
    "models/feature_columns.pkl",
    "models/cyber_model_online.pkl",
    "reports/metrics.json",
    "reports/prequential_results.json",
    "reports/drift_detection_results.json",
    "reports/classification_report.txt",
    "reports/confusion_matrix.csv",
    "reports/classification_metrics.csv",
    "reports/prequential_batch_results.csv",
]

print("\n========== FILE CHECK ==========")
for filename in required_files:
    path = ROOT / filename
    print(f"{'OK' if path.exists() else 'MISSING'}  {filename}")

def show_json(filename):
    path = ROOT / filename
    print(f"\n========== {filename} ==========")
    if not path.exists():
        print("File not found.")
        return
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(json.dumps(data, indent=2, default=str))
    except Exception as error:
        print("Could not read JSON:", error)

show_json("reports/metrics.json")
show_json("reports/prequential_results.json")
show_json("reports/drift_detection_results.json")

print("\n========== DATASET CHECK ==========")
cleaned_path = ROOT / "data/processed/MedSec-25-cleaned.csv"

if cleaned_path.exists():
    df = pd.read_csv(cleaned_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", int(df.isna().sum().sum()))

    numeric = df.select_dtypes(include="number").drop(
        columns=["Label"], errors="ignore"
    )
    negative_counts = (numeric < 0).sum()
    negative_counts = negative_counts[negative_counts > 0]

    print("\nNumeric columns containing negative values:")
    if negative_counts.empty:
        print("None")
    else:
        print(negative_counts.to_string())

print("\n========== AUDIT FINISHED ==========")
