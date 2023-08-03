from mmengine import read_base
from DVM import DVMTrainner

with read_base():
    from ._base_.train_dataset import train_dataset
    from ._base_.unet import unet
    from ._base_.noise_scheduler import noise_scheduler

device = "cuda"
train_config = dict(
    device=device,
    output_dir="caches/ventilation_v1/unet",
    prediction_point=32,
    num_train_timesteps=200,
    num_epochs=1000,
    train_batch_size=1500,
    eval_batch_size=1,
    learning_rate=0.001,
    lr_warmup_steps=100,
    eval_interval=100,
)

trainner = dict(
    type=DVMTrainner,
    DVM_config=train_config,
    u_net=unet,
    noise_scheduler=noise_scheduler,
    optimizer=dict(type="Adam", lr=0.001),
    dataset=train_dataset, )

wandb_config = dict()
