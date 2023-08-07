from mmengine import read_base
from DVM import DVMEvaler,Long_predictionVentilationDataset

with read_base():
    from ._base_.train_dataset import train_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"

dataset = dict(
    type=Long_predictionVentilationDataset,
    csv_path='data/v1.csv',
)

train_config = dict(
    device=device,
    output_dir="caches/ventilation_v1/unet",
    u_net_weight_path = None,
    prediction_point=32,
    num_train_timesteps=200,
    batch_size=10,
    batch_index=3,
)

project_name = prject_name

evaler = dict(
    type=DVMEvaler,
    evaler_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    dataset=train_dataset,)
