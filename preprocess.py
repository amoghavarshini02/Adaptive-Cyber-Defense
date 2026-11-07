 
import pandas as pd
from pathlib import Path

# File paths
RAW_PATH = Path("data/raw/MedSec-25.csv")
OUTPUT_PATH = Path("data/processed/MedSec-25-cleaned.csv")

print("Loading MedSec-25 dataset...")
df = pd.read_csv(RAW_PATH)

original_rows = len(df)
print("Original records:", original_rows)

# 1. Parse timestamps
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

invalid_timestamps = df["Timestamp"].isna().sum()
df = df.dropna(subset=["Timestamp"])

# 2. Remove physically impossible negative measurements.
# Init Bwd Win Byts is deliberately excluded:
# -1 is preserved as a protocol-specific sentinel.

nonnegative_features = [
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Max",
    "Bwd IAT Min"
]

negative_mask = (
    df[nonnegative_features] < 0
).any(axis=1)

invalid_measurement_rows = int(negative_mask.sum())

df = df.loc[~negative_mask].copy()

# 3. Sort chronologically
df = df.sort_values(
    by="Timestamp"
).reset_index(drop=True)

# 4. Drop identifier columns
identifier_columns = [
    "Flow ID",
    "Src IP",
    "Dst IP"
]

df = df.drop(
    columns=identifier_columns,
    errors="ignore"
)

# 5. Drop constant columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
    and col != "Label"
]

df = df.drop(columns=constant_columns)

# 6. Replace infinite numeric values and handle missing values
numeric_columns = df.select_dtypes(
    include="number"
).columns

df[numeric_columns] = df[numeric_columns].replace(
    [float("inf"), float("-inf")],
    float("nan")
)

df[numeric_columns] = df[numeric_columns].fillna(0)

# 7. Encode target labels
label_mapping = {
    "Benign": 0,
    "Exfiltration": 1,
    "Initial access": 2,
    "Lateral movement": 3,
    "Reconnaissance": 4
}

df["Label"] = df["Label"].map(label_mapping)

if df["Label"].isna().any():
    raise ValueError("Unknown or missing labels found!")

df["Label"] = df["Label"].astype(int)

# 8. Save processed dataset
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

# 9. Final summary
print("\n========== PREPROCESSING SUMMARY ==========")
print("Original records:", original_rows)
print("Invalid timestamps removed:", invalid_timestamps)
print("Negative-measurement records removed:",
      invalid_measurement_rows)
print("Final records:", len(df))
print("Final columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))
print("Constant columns removed:", constant_columns)

print("\nLabel distribution:")
print(df["Label"].value_counts().sort_index())

print("\nInit Bwd Win Byts sentinel values preserved:")
if "Init Bwd Win Byts" in df.columns:
    print(
        int((df["Init Bwd Win Byts"] == -1).sum())
    )

print("\nSaved to:", OUTPUT_PATH)
print("Preprocessing completed successfully!")
