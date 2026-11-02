"""
Just for loading label names.
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