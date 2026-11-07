
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from river.drift import ADWIN

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
# LOAD DATASET AND BASELINE MODEL
# --------------------------------------------------

print("Loading dataset and model...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Label", "Timestamp"], errors="ignore")
y = df["Label"].astype(int)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_columns = joblib.load(FEATURE_FILE)

X = X[feature_columns]

# Reproduce the same stratified test stream.
_, X_stream, _, y_stream = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_stream = X_stream.reset_index(drop=True)
y_stream = y_stream.reset_index(drop=True)

X_scaled = scaler.transform(X_stream)

# --------------------------------------------------
# INITIALIZE ADWIN
# --------------------------------------------------

print("\nInitializing ADWIN drift detector...")

drift_detector = ADWIN(
    delta=0.002,
    clock=32
)

drift_events = []
total_records = len(X_scaled)

# --------------------------------------------------
# MONITOR PREDICTION ERRORS
# --------------------------------------------------

print("\nMonitoring incoming batches...")

for start in range(0, total_records, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_records)

    X_batch = X_scaled[start:end]
    y_batch = y_stream.iloc[start:end].to_numpy()

    predictions = model.predict(X_batch)

    batch_errors = (predictions != y_batch).astype(int)

    # Feed each prediction error to ADWIN.
    for offset, error in enumerate(batch_errors):

        drift_detector.update(int(error))

        if drift_detector.drift_detected:

            event = {
                "record_number": start + offset + 1,
                "batch_number": (start // BATCH_SIZE) + 1,
                "estimated_error_rate": float(drift_detector.estimation),
                "detector_width": int(drift_detector.width)
            }

            drift_events.append(event)

            print(
                "\nPOTENTIAL CONCEPT DRIFT DETECTED!"
                f"\nRecord: {event['record_number']:,}"
                f"\nBatch: {event['batch_number']}"
                f"\nEstimated error rate: "
                f"{event['estimated_error_rate']:.4f}"
            )

    # Update the model after evaluating this batch.
    model.partial_fit(
        X_batch,
        y_batch,
        classes=np.array(sorted(y.unique()))
    )

    print(
        f"Monitored {end:,}/{total_records:,} records | "
        f"Drift events: {len(drift_events)}"
    )

# --------------------------------------------------
# SAVE DRIFT RESULTS
# --------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "detector": "ADWIN",
    "delta": 0.002,
    "batch_size": BATCH_SIZE,
    "records_monitored": total_records,
    "drift_events_detected": len(drift_events),
    "events": drift_events
}

with open(
    REPORT_DIR / "drift_detection_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\n========== DRIFT DETECTION SUMMARY ==========")
print("Detector: ADWIN")
print("Records monitored:", f"{total_records:,}")
print("Drift events detected:", len(drift_events))
print("Results saved to:", REPORT_DIR / "drift_detection_results.json")

print("\nDRIFT DETECTION COMPLETED!")
