import os, errno
import cv2
import torch
import numpy as np

from glob import glob
from PIL import Image
from torch.autograd import Variable
from albumentations.pytorch.functional import img_to_tensor

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

@torch.no_grad()
def predict_from_image(model, path_images, path_out, threshold=0.5):
    create_dir(path_out)
    for path_image in glob(path_images+"*.png") + glob(path_images+"*.jpg"):
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_tensor(image)
        image = Variable(image.unsqueeze(0).float())

        prediction = model(image)
        prediction = prediction.squeeze(0)
        prediction = (prediction>threshold).float()*1
        prediction = prediction.numpy().astype(np.uint32) * 255

        pil_im = Image.fromarray(prediction[0]).convert('RGB')
        pil_im = pil_im.resize((image.shape[2], image.shape[3]))
        pil_im.save(os.path.join(path_out, os.path.basename(path_image)))