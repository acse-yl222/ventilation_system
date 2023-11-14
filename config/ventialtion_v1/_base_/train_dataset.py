from DVM import VentilationDataset

train_dataset = dict(
    type=VentilationDataset,
    csv_path="data/train_1.csv",
)

val_dataset = dict(
    type=VentilationDataset,
    csv_path="data/val_1.csv",
)