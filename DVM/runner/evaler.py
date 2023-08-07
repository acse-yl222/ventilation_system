from mmengine import Config
from ..registry import DVMR, DVDR
from torchvision.transforms import Compose, Normalize
from ..pipeline import DDIMPipeline
from torch.utils.data import DataLoader
import torch
from diffusers.utils import randn_tensor
import numpy as np
from ..utils import compare
from ..registry import DVMR, DVDR


@DVMR.register_module()
class DVMEvaler:
    def __init__(self,
                 unet: Config,
                 dataset: Config,
                 evaler_config: Config,
                 noise_scheduler: Config):
        self.unet = DVMR.build(unet)
        self.noise_scheduler = DVMR.build(noise_scheduler)
        self.dataset = DVDR.build(dataset)
        self.config = evaler_config
        data_title_string = "time_sin,time_cos,Occupancy,door_gap,window_gap,humidity,VOC_ppb,temperature_Main,temperature_FRT,temperature_FRM,temperature_FRB,temperature_FMT,temperature_FMM,temperature_FMB,temperature_FLT,temperature_FLM,temperature_FLB,temperature_BRT,temperature_BRM,temperature_BRB,temperature_BMT,temperature_BMM,temperature_BMB,temperature_BLT,temperature_BLM,temperature_BLB,temperature_WRB,temperature_WMB,temperature_WLB,temperature_WLF,temperature_DoorRT,temperature_BTable,temperature_PRUR,temperature_PRUL,temperature_PRDR,temperature_PRDL,temperature_PLDR,temperature_PLDL,temperature_Out,light_BLM,light_BLB,light_WRB,light_WMB,light_WLB,light_WLF,light_DoorRT,light_BTable,light_PRUR,light_PRUL,light_PRDR,light_PRDL,light_PLDR,light_PLDL,light_Out,outdoor_temperature,outdoor_humidity,outdoor_windgust,outdoor_windspeed,outdoor_winddir,outdoor_sealevelpressure,outdoor_dew,outdoor_cloudcover,outdoor_solarradiation,outdoor_solarenergy"
        self.title_list = data_title_string.split(',')
        self.pipeline = DDIMPipeline(unet=self.unet, scheduler=noise_scheduler)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.config.batch_size, shuffle=True)
        self.list1 = ["Occupancy", "door_gap", "window_gap", "humidity",
                   "VOC_ppm", "temperature_Main", "outdoor_temperature", "outdoor_windgust", "outdoor_humidity", ]


    def initial_unet(self):
        self.u_net.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.u_net.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print("Load u_net weight from {}".format(self.config.u_net_weight_path))
        else:
            print("No u_net weight path is provided, use random weight")

    def get_batch(self,index):
        # select desired image to do the test.
        data_iter = iter(self.dataloader)
        desired_index = 300

        try:
            for i in range(desired_index + 1):
                batch = next(data_iter)
        except StopIteration:
            print("索引超出了DataLoader中的batch数量。")

        # precess the data to thr prediction
        inputs = batch[0]
        return inputs

    def long_time_eval(self):
        inputs = self.get_batch(self.config.batch_index)
        inputs.to(self.config.device)
        image = randn_tensor(inputs.shape, device=self.config.device, dtype=self.unet.dtype)
        image[:, :, 0:self.config.prediction_point, :] = inputs[:, :, 0:self.config.prediction_point, :]

        # now the image is half clear image half randon noise.

        # use 1 iteration scheme to do the prediction.
        for i in range(1):
            image = self.pipeline.do_prediction(
                image,
                self.config.prediction_point,
                batch_size=len(inputs),
                num_inference_steps=self.config.num_train_timesteps,
                output_type='numpy'
            )
        # post process the images to make it to 0-1
        predicted_image = (image / 2 + 0.5).clamp(0, 1)
        predicted_image = predicted_image.cpu().permute(0, 2, 3, 1).numpy()

        original_image = (inputs / 2 + 0.5).clamp(0, 1)
        original_image = original_image.cpu().permute(0, 2, 3, 1).numpy()

        # recover the value to original value scale
        min_number = np.zeros_like(predicted_image)
        max_number = np.zeros_like(predicted_image)
        for i in range(len(predicted_image)):
            for row in range(len(predicted_image[0])):
                min_number[i, row, :, 0] = self.dataset.min_value
                max_number[i, row, :, 0] = self.dataset.max_value
        predicted_image = predicted_image * (max_number - min_number) + min_number
        original_image = original_image * (max_number - min_number) + min_number
        predicted_image_ = predicted_image[0, :, :, 0]
        original_image_ = original_image[0, :, :, 0]
        compare(original_image_, predicted_image_, self.config.prediction_point, self.title_list)