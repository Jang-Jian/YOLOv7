import os
import torch


def configure_inference_device(device_name: str):
    _device_name = device_name.lower()
    device_info = ""
    if _device_name == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        device_info = "cpu"
    elif _device_name:
        os.environ['CUDA_VISIBLE_DEVICES'] = _device_name
        if torch.cuda.is_available():
            device_info = "cuda:" + str(_device_name)
        else:    
            return None, "CUDA device is not being found."

    if device_info != "":
        return torch.device(device_info), ""
    
    return None, "The device is not being allocated."