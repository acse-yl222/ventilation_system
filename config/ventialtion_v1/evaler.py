from mmengine import read_base
from DVM import DVMEvaler

with read_base():
    from ._base_.train_dataset import train_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"
train_config = dict(
    device=device,
    output_dir="caches/ventilation_v1/unet",
    u_net_weight_path = None,
    prediction_point=32,
    num_train_timesteps=200,
    num_epochs=1000,
    train_batch_size=10,
    eval_batch_size=1,
    learning_rate=0.001,
    lr_warmup_steps=100,
    eval_interval=100,
)

project_name = prject_name

evaler = dict(
    type=DVMEvaler,
    trainner_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optimizer=dict(type="Adam", learning_rate=0.001),
    dataset=train_dataset,)
