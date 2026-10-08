import pandas as pd

train = pd.read_csv("ml/data/splits/train.csv")
val = pd.read_csv("ml/data/splits/val.csv")

merged = pd.concat([train, val], ignore_index=True)

merged.to_csv("ml/data/splits/train_val.csv", index=False)