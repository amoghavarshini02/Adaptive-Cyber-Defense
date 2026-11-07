
import pandas as pd

df = pd.read_csv("data/processed/MedSec-25-cleaned.csv")

invalid_features = [
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

invalid_mask = (df[invalid_features] < 0).any(axis=1)
sentinel_mask = df["Init Bwd Win Byts"] == -1

print("TOTAL RECORDS:", len(df))
print("Rows with physically invalid negative measurements:",
      int(invalid_mask.sum()))
print("Rows with Init Bwd Win Byts = -1:",
      int(sentinel_mask.sum()))
print("Rows containing either condition:",
      int((invalid_mask | sentinel_mask).sum()))

print("\nINVALID MEASUREMENTS BY CLASS:")
print(df.loc[invalid_mask, "Label"].value_counts().sort_index())

print("\nINIT BWD WIN BYTES SENTINEL BY CLASS:")
print(df.loc[sentinel_mask, "Label"].value_counts().sort_index())

print("\nBACKWARD TRAFFIC COLUMNS AVAILABLE:")
candidates = [
    "Tot Bwd Pkts",
    "Total Bwd Packets",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Mean",
    "Bwd Pkts/s"
]

available = [c for c in candidates if c in df.columns]
print(available)

if available:
    print("\nSAMPLE SENTINEL RECORDS:")
    print(
        df.loc[sentinel_mask, ["Init Bwd Win Byts"] + available + ["Label"]]
        .head(10)
        .to_string(index=False)
    )

print("\nDIAGNOSTIC COMPLETE")
