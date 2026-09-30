"""Generate sample_transactions.csv from the real training data (10 rows)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

tx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_transaction.csv"), nrows=200, low_memory=False)
idt = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_identity.csv"), low_memory=False)
df = tx.merge(idt, on="TransactionID", how="left")

# Pick 10 rows — 8 legit + 2 fraud to show diversity
legit = df[df["isFraud"] == 0].head(8)
fraud = df[df["isFraud"] == 1].head(2)
sample = pd.concat([legit, fraud]).reset_index(drop=True)

# Keep only the most important columns (not all 434 — keep it readable)
# But include enough for meaningful graph construction and prediction
keep_cols = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "V1", "V2", "V3", "V4", "V5", "V12", "V13", "V14", "V15",
    "V44", "V45", "V53", "V54", "V55", "V56", "V75", "V76", "V77", "V78",
    "V126", "V127", "V128", "V129", "V130", "V131", "V279", "V280",
    "V306", "V307", "V308", "V309", "V310", "V311", "V312", "V313", "V314",
    "V315", "V316", "V317",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31", "id_33",
    "id_34", "id_35", "id_36", "id_37", "id_38",
]

# Only keep columns that actually exist
keep_cols = [c for c in keep_cols if c in sample.columns]
sample = sample[keep_cols]

sample.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_transactions.csv"), index=False)
print(f"Generated sample_transactions.csv: {len(sample)} rows x {len(sample.columns)} columns")
print(f"Columns: {list(sample.columns)[:20]}...")
print(f"Fraud count: {sample['isFraud'].sum()}")
