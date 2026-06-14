"""
make_label_mapping.py
=====================
Generate the integer->label mapping used to train the supervised models, so the
app can translate a numeric prediction (0..k-1) back into the readable profile
name (e.g. 'light acidic reds').

The mapping MUST match the notebook: sorted unique labels -> 0..k-1.
"""

import json
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/processed/wine_labeled.csv")
classes = sorted(df["cluster_label"].unique())
int_to_label = {i: name for i, name in enumerate(classes)}

Path("models").mkdir(parents=True, exist_ok=True)
with open("models/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(int_to_label, f, ensure_ascii=False, indent=2)

print("Saved models/label_mapping.json:")
print(json.dumps(int_to_label, ensure_ascii=False, indent=2))
