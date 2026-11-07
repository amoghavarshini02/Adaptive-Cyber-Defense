import pandas as pd

# Dataset path
file_path = "data/raw/MedSec-25.csv"


# 1. LOAD DATASET PREVIEW
print("\n========== DATASET PREVIEW ==========")

df = pd.read_csv(file_path, nrows=5)

print(df.head())


# 2. DISPLAY COLUMN NAMES
print("\n========== COLUMN NAMES ==========")

print(df.columns.tolist())


# 3. DISPLAY DATA TYPES
print("\n========== DATA TYPES ==========")

print(df.dtypes)


# 4. ANALYSE ATTACK CATEGORIES
print("\n========== ATTACK CATEGORIES ==========")

label_counts = {}

for chunk in pd.read_csv(
    file_path,
    usecols=["Label"],
    chunksize=50000
):
    counts = chunk["Label"].value_counts()

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count

for label, count in sorted(label_counts.items()):
    print(f"{label}: {count:,}")


# 5. TOTAL RECORDS
total_records = sum(label_counts.values())

print("\n========== DATASET SUMMARY ==========")
print(f"Total records: {total_records:,}")
print(f"Total columns: {len(df.columns)}")
print(f"Total attack categories: {len(label_counts)}")

print("\nDataset inspection completed!")
print("\n========== MISSING VALUES ==========")

df_full = pd.read_csv(file_path)

print(df_full.isnull().sum().sum())
print("\n========== DUPLICATE RECORDS ==========")

print(df_full.duplicated().sum())
import numpy as np

print("\n========== INFINITE VALUES ==========")

numeric_df = df_full.select_dtypes(include="number")

print(np.isinf(numeric_df).sum().sum())

import pandas as pd
import numpy as np

# Dataset path
file_path = "data/raw/MedSec-25.csv"

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(file_path)

# 1. BASIC DATASET INFORMATION
print("\n========== DATASET OVERVIEW ==========")
print("Total records:", f"{len(df):,}")
print("Total columns:", len(df.columns))
print("Memory usage:", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# 2. COLUMN NAMES AND DATA TYPES
print("\n========== COLUMN NAMES AND DATA TYPES ==========")
print(df.dtypes.to_string())

# 3. TARGET LABEL DISTRIBUTION
print("\n========== LABEL DISTRIBUTION ==========")
print(df["Label"].value_counts().to_string())

print("\n========== LABEL PERCENTAGES ==========")
print((df["Label"].value_counts(normalize=True) * 100).round(2).to_string())

# 4. MISSING VALUES
print("\n========== MISSING VALUES ==========")
print("Total missing values:", df.isnull().sum().sum())

missing = df.isnull().sum()
print(missing[missing > 0].to_string() if (missing > 0).any() else "No missing values found.")

# 5. DUPLICATE RECORDS
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

# 6. INFINITE VALUES
print("\n========== INFINITE VALUES ==========")
numeric_df = df.select_dtypes(include="number")
print("Total infinite values:", np.isinf(numeric_df).sum().sum())

# 7. TIMESTAMP VALIDATION
print("\n========== TIMESTAMP CHECK ==========")
timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
print("Original timestamp data type:", df["Timestamp"].dtype)
print("Invalid timestamps:", timestamps.isna().sum())
print("Earliest timestamp:", timestamps.min())
print("Latest timestamp:", timestamps.max())
print("Chronologically sorted:", timestamps.is_monotonic_increasing)

# 8. NUMERIC FEATURE SUMMARY
print("\n========== NUMERIC FEATURE SUMMARY ==========")
print(numeric_df.describe().transpose().to_string())

# 9. CONSTANT COLUMNS
print("\n========== CONSTANT COLUMNS ==========")
constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]
print(constant_columns if constant_columns else "No constant columns found.")

# 10. UNIQUE VALUES IN CATEGORICAL COLUMNS
print("\n========== CATEGORICAL FEATURE CHECK ==========")
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} unique values")

# 11. FINAL SUMMARY
print("\n========== DATASET INSPECTION COMPLETED ==========")
print("Records:", f"{len(df):,}")
print("Features:", len(df.columns) - 1)
print("Target column: Label")
print("Attack categories:", df["Label"].nunique())
print("Inspection completed successfully!")
