from mmengine import read_base
from DVM import DVMTrainner

with read_base():
    from ._base_.train_dataset import train_dataset, val_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"
train_config = dict(
    device=device,
    output_dir="caches/ventilation_v4/unet",
    u_net_weight_path="caches/ventilation_v3/unet/31000/unet.pth",
    prediction_point=42,
    num_train_timesteps=200,
    num_epochs=12,
    train_batch_size=20,
    eval_batch_size=40,
    learning_rate=0.00005,
    lr_warmup_steps=10000,
    eval_begin=10000,
    eval_interval=5000,
)

project_name = prject_name

trainner = dict(
    type=DVMTrainner,
    trainner_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optimizer=dict(type="Adam", learning_rate=0.00005),
    train_dataset=train_dataset,
    val_dataset=val_dataset)

wandb_config = dict(
    learning_rate=trainner['optimizer']['learning_rate'],
    architecture="diffusion model",
    dataset=project_name,
    epochs=train_config['num_epochs'],
)