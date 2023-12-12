import cv2
import torch

from model.unet import UNet, DecoderBlock, Conv2dReLU
from utils import predict_from_image

path_model =""
path_im = ""
path_out = ""

model = UNet()
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu'))["model"].state_dict())

predict_from_image(model, path_im, path_out)