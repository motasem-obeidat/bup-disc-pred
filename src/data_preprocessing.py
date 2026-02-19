import os
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "BUP_STATUS_NUM"
SEED = 42

# Load dataset
df = pd.read_excel("/bup_features.xlsx")

# Create output directory
os.makedirs("/dataset", exist_ok=True)

# Initialize datasets
df_multi = df.copy()
df_binary = df.copy()

# Binarize target
df_binary[TARGET] = df_binary[TARGET].apply(lambda x: 1 if x in [1, 2] else 0)

# Split 1: Train (70%) and Temp (30%)
train_idx, temp_idx = train_test_split(
    df_multi.index,
    test_size=0.3,
    random_state=SEED,
    shuffle=True,
    stratify=df_multi[TARGET],
)

# Split 2: Val (15%) and Test (15%) from Temp
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=SEED,
    shuffle=True,
    stratify=df_multi.loc[temp_idx, TARGET],
)

# Apply splits
train_multi = df_multi.loc[train_idx]
val_multi = df_multi.loc[val_idx]
test_multi = df_multi.loc[test_idx]

train_binary = df_binary.loc[train_idx]
val_binary = df_binary.loc[val_idx]
test_binary = df_binary.loc[test_idx]

# Save datasets
train_multi.to_csv("/dataset/train_multi.csv", index=False)
val_multi.to_csv("/dataset/val_multi.csv", index=False)
test_multi.to_csv("/dataset/test_multi.csv", index=False)

train_binary.to_csv("/dataset/train_binary.csv", index=False)
val_binary.to_csv("/dataset/val_binary.csv", index=False)
test_binary.to_csv("/dataset/test_binary.csv", index=False)

print(f"Stratified split complete.")
