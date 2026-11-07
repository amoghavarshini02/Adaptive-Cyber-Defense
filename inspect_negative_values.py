import pandas as pd

df = pd.read_csv("data/processed/MedSec-25-cleaned.csv")

columns = [
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
    "Bwd IAT Min",
    "Init Bwd Win Byts"
]

for col in columns:
    if col not in df.columns:
        continue

    negative = df[df[col] < 0]

    if not negative.empty:
        print(f"\n{'=' * 55}")
        print(f"FEATURE: {col}")
        print("Negative records:", len(negative))
        print("Minimum value:", negative[col].min())
        print("Maximum negative value:", negative[col].max())

        display_columns = [
            c for c in ["Timestamp", col, "Label"]
            if c in df.columns
        ]

        print(negative[display_columns].head(5).to_string(index=False))