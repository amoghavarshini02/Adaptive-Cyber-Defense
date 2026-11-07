
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_FILE = "data/processed/MedSec-25-cleaned.csv"
MODEL_FILE = "models/cyber_model.pkl"
SCALER_FILE = "models/scaler.pkl"
FEATURE_FILE = "models/feature_columns.pkl"

REPORT_DIR = Path("reports")
BATCH_SIZE = 5000
RANDOM_STATE = 42

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading dataset and baseline model...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Label", "Timestamp"], errors="ignore")
y = df["Label"].astype(int)

# --------------------------------------------------
# RECREATE THE SAME STRATIFIED SPLIT
# --------------------------------------------------

X_initial, X_stream, y_initial, y_stream = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------------------------------------
# LOAD A FRESH BASELINE MODEL
# --------------------------------------------------

# Load a fresh copy for this experiment.
# Do not reuse a model already updated in a previous run.

baseline_model = joblib.load(MODEL_FILE)
model = joblib.load(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)
feature_columns = joblib.load(FEATURE_FILE)

X_stream = X_stream[feature_columns].reset_index(drop=True)
y_stream = y_stream.reset_index(drop=True)

X_stream_scaled = scaler.transform(X_stream)

classes = np.array(sorted(y.unique()))

print("Incoming stream records:", f"{len(X_stream):,}")
print("Batch size:", BATCH_SIZE)

# --------------------------------------------------
# PREQUENTIAL EVALUATION
# --------------------------------------------------

print("\nStarting prequential evaluation...")
print("Process: Predict -> Evaluate -> Learn")

batch_results = []

all_predictions = []
all_actuals = []

total_records = len(X_stream)

for start in range(0, total_records, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_records)

    X_batch = X_stream_scaled[start:end]
    y_batch = y_stream.iloc[start:end].to_numpy()

    # STEP 1: Predict before learning from this batch.
    predictions = model.predict(X_batch)

    # STEP 2: Evaluate predictions.
    batch_accuracy = accuracy_score(
        y_batch,
        predictions
    )

    batch_balanced_accuracy = balanced_accuracy_score(
        y_batch,
        predictions
    )

    batch_macro_f1 = f1_score(
        y_batch,
        predictions,
        average="macro",
        zero_division=0
    )

    all_predictions.extend(predictions)
    all_actuals.extend(y_batch)

    # STEP 3: Store metrics before updating.
    batch_results.append({
        "batch": len(batch_results) + 1,
        "records_processed": end,
        "batch_size": end - start,
        "accuracy": float(batch_accuracy),
        "balanced_accuracy": float(batch_balanced_accuracy),
        "macro_f1": float(batch_macro_f1)
    })

    print(
        f"Batch {len(batch_results):02d} | "
        f"Records: {end:,}/{total_records:,} | "
        f"Accuracy: {batch_accuracy * 100:.2f}% | "
        f"Macro F1: {batch_macro_f1 * 100:.2f}%"
    )

    # STEP 4: Learn only after evaluating the batch.
    model.partial_fit(
        X_batch,
        y_batch,
        classes=classes
    )

# --------------------------------------------------
# AGGREGATE PREQUENTIAL PERFORMANCE
# --------------------------------------------------

prequential_accuracy = accuracy_score(
    all_actuals,
    all_predictions
)

prequential_balanced_accuracy = balanced_accuracy_score(
    all_actuals,
    all_predictions
)

prequential_macro_f1 = f1_score(
    all_actuals,
    all_predictions,
    average="macro",
    zero_division=0
)

# --------------------------------------------------
# BASELINE COMPARISON
# --------------------------------------------------

baseline_predictions = baseline_model.predict(X_stream_scaled)

baseline_accuracy = accuracy_score(
    y_stream,
    baseline_predictions
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_stream,
    baseline_predictions
)

baseline_macro_f1 = f1_score(
    y_stream,
    baseline_predictions,
    average="macro",
    zero_division=0
)

print("\n========== BASELINE PERFORMANCE ==========")
print(f"Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {baseline_balanced_accuracy * 100:.2f}%")
print(f"Macro F1: {baseline_macro_f1 * 100:.2f}%")

print("\n========== PREQUENTIAL PERFORMANCE ==========")
print(f"Accuracy: {prequential_accuracy * 100:.2f}%")
print(
    f"Balanced Accuracy: "
    f"{prequential_balanced_accuracy * 100:.2f}%"
)
print(f"Macro F1: {prequential_macro_f1 * 100:.2f}%")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "experiment": "Prequential test-then-train evaluation",
    "batch_size": BATCH_SIZE,
    "total_stream_records": total_records,
    "baseline": {
        "accuracy": float(baseline_accuracy),
        "balanced_accuracy": float(baseline_balanced_accuracy),
        "macro_f1": float(baseline_macro_f1)
    },
    "prequential": {
        "accuracy": float(prequential_accuracy),
        "balanced_accuracy": float(prequential_balanced_accuracy),
        "macro_f1": float(prequential_macro_f1)
    },
    "batch_results": batch_results
}

with open(
    REPORT_DIR / "prequential_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

pd.DataFrame(batch_results).to_csv(
    REPORT_DIR / "prequential_batch_results.csv",
    index=False
)

# Save the updated experimental model separately.
joblib.dump(
    model,
    "models/cyber_model_online.pkl"
)

print("\nResults saved:")
print("reports/prequential_results.json")
print("reports/prequential_batch_results.csv")
print("models/cyber_model_online.pkl")

print("\nPREQUENTIAL EVALUATION COMPLETED!")
