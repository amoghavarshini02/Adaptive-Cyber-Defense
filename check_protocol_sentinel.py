import pandas as pd

df = pd.read_csv("data/processed/MedSec-25-cleaned.csv")

print("Protocol distribution:")
print(df["Protocol"].value_counts(dropna=False))

print("\nInit Bwd Win Byts = -1 by protocol:")
sentinel = df[df["Init Bwd Win Byts"] == -1]

print(
    sentinel.groupby("Protocol")
    .agg(
        records=("Label", "size"),
        classes=("Label", "nunique")
    )
    .to_string()
)

print("\nAll protocols and sentinel proportions:")
summary = df.groupby("Protocol").agg(
    total_records=("Label", "size"),
    sentinel_records=(
        "Init Bwd Win Byts",
        lambda x: (x == -1).sum()
    )
)
summary["sentinel_percentage"] = (
    summary["sentinel_records"] / summary["total_records"] * 100
)

print(summary.to_string())