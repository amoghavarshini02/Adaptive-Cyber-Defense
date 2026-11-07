 
import json
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_FILE = "data/processed/MedSec-25-cleaned.csv"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

BATCH_SIZE = 5000
TEST_SIZE = 0.20
RANDOM_STATE = 42

LABEL_NAMES = {
    0: "Benign",
    1: "Exfiltration",
    2: "Initial access",
    3: "Lateral movement",
    4: "Reconnaissance"
}

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading processed dataset...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Label", "Timestamp"], errors="ignore")
y = df["Label"].astype(int)

print("Total records:", f"{len(df):,}")
print("Total features:", X.shape[1])

# --------------------------------------------------
# STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

classes = np.array(sorted(y.unique()))

print("\nTraining records:", f"{len(X_train):,}")
print("Testing records:", f"{len(X_test):,}")

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = dict(zip(classes, class_weights))

# --------------------------------------------------
# INCREMENTAL MODEL TRAINING
# --------------------------------------------------

print("\nTraining model incrementally...")

model = SGDClassifier(
    loss="log_loss",
    learning_rate="optimal",
    random_state=RANDOM_STATE
)

total_records = len(X_train_scaled)

for start in range(0, total_records, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_records)

    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end].to_numpy()

    sample_weights = np.array([
        class_weight_dict[label] for label in y_batch
    ])

    model.partial_fit(
        X_batch,
        y_batch,
        classes=classes,
        sample_weight=sample_weights
    )

    print(f"Processed {end:,} / {total_records:,} records")

print("\nInitial training completed!")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=classes,
    target_names=[LABEL_NAMES[c] for c in classes],
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_test,
    y_pred,
    labels=classes,
    target_names=[LABEL_NAMES[c] for c in classes],
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=classes)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Macro F1 Score: {macro_f1 * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(report_text)

print("\n========== CONFUSION MATRIX ==========")

confusion_df = pd.DataFrame(
    cm,
    index=[LABEL_NAMES[c] for c in classes],
    columns=[LABEL_NAMES[c] for c in classes]
)

print(confusion_df)

# --------------------------------------------------
# SAVE MODEL FILES
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_DIR / "cyber_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")

# --------------------------------------------------
# SAVE EVALUATION REPORTS
# --------------------------------------------------

metrics = {
    "accuracy": round(float(accuracy), 4),
    "balanced_accuracy": round(float(balanced_accuracy), 4),
    "macro_f1": round(float(macro_f1), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "features": int(X.shape[1]),
    "batch_size": BATCH_SIZE,
    "classes": LABEL_NAMES
}

with open(REPORT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(REPORT_DIR / "classification_report.txt", "w") as file:
    file.write(report_text)

confusion_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

pd.DataFrame(report_dict).transpose().to_csv(
    REPORT_DIR / "classification_metrics.csv"
)

print("\n========== FILES SAVED ==========")
print("Model:", MODEL_DIR / "cyber_model.pkl")
print("Scaler:", MODEL_DIR / "scaler.pkl")
print("Feature list:", MODEL_DIR / "feature_columns.pkl")
print("Metrics:", REPORT_DIR / "metrics.json")
print("Classification report:", REPORT_DIR / "classification_report.txt")
print("Confusion matrix:", REPORT_DIR / "confusion_matrix.csv")
print("Class metrics:", REPORT_DIR / "classification_metrics.csv")

print("\nMODEL TRAINING AND REPORT GENERATION COMPLETED!")
