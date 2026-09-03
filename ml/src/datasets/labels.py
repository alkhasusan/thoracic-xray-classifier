"""
Utilities for handling disease labels in the NIH ChestX-ray14 dataset.
"""

DISEASE_LABELS = [
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

NUM_CLASSES = len(DISEASE_LABELS)


def get_disease_labels():
    """
    Return the list of disease labels.
    """
    return DISEASE_LABELS.copy()