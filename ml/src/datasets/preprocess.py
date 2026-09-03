"""
preprocess.py

Preprocesses the NIH ChestX-ray14 dataset.

This script:
1. Locates the dataset
2. Loads the metadata CSV
3. Converts disease labels into binary columns
4. Matches each image with its full file path
5. Splits the dataset by Patient ID
6. Saves train.csv, val.csv and test.csv


from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# Dataset Paths
# =====================================================

# Kaggle dataset root
KAGGLE_DATASET_PATH = Path(
    "/kaggle/input/datasets/biditdas06/nih-chestxray14"
)

# Metadata CSV
METADATA_PATH = KAGGLE_DATASET_PATH / "Data_Entry_2017_v2020.csv"

# =====================================================
# Disease Labels
# =====================================================

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# =====================================================
# Load Metadata
# =====================================================

print("=" * 60)
print("Loading Metadata")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)

print(f"Loaded {len(df)} records.")
print()

# =====================================================
# Convert Disease Labels to Binary Columns
# =====================================================

print("=" * 60)
print("Encoding Disease Labels")
print("=" * 60)

for disease in DISEASES:
    df[disease] = df["Finding Labels"].apply(
        lambda labels: int(disease in labels.split("|"))
    )

print("Disease labels encoded successfully.")
print()

# =====================================================
# Locate Image Files
# =====================================================

print("=" * 60)
print("Locating Images")
print("=" * 60)

image_paths = {}

for image in KAGGLE_DATASET_PATH.glob("images_*/images/*.png"):
    image_paths[image.name] = str(image)

print(f"Found {len(image_paths)} images.")

# Add image path column
df["Image Path"] = df["Image Index"].map(image_paths)

print("Image paths matched successfully.")
print()

# =====================================================
# Patient-wise Train / Validation / Test Split
# =====================================================

print("=" * 60)
print("Creating Patient-wise Split")
print("=" * 60)

patients = df["Patient ID"].unique()

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=42,
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
)

train_df = df[df["Patient ID"].isin(train_patients)]
val_df = df[df["Patient ID"].isin(val_patients)]
test_df = df[df["Patient ID"].isin(test_patients)]

print(f"Train images      : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Test images       : {len(test_df)}")
print()

# =====================================================
# Verify No Data Leakage
# =====================================================

print("=" * 60)
print("Checking Patient Leakage")
print("=" * 60)

assert len(set(train_df["Patient ID"]) & set(val_df["Patient ID"])) == 0
assert len(set(train_df["Patient ID"]) & set(test_df["Patient ID"])) == 0
assert len(set(val_df["Patient ID"]) & set(test_df["Patient ID"])) == 0

print("No patient leakage detected.")
print()

# =====================================================
# Save CSV Files
# =====================================================

print("=" * 60)
print("Saving CSV Files")
print("=" * 60)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Saved:")
print(" - train.csv")
print(" - val.csv")
print(" - test.csv")
print()

print("=" * 60)
print("Preprocessing Complete!")
print("=" * 60)

"""