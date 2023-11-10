from DVM import VentilationDataset

train_dataset = dict(
    type=VentilationDataset,
    csv_path="data/train.csv",
)

val_dataset = dict(
    type=VentilationDataset,
    csv_path="data/val.csv",
)