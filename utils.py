import os
import cv2
import torch
import numpy as np

from glob import glob
from PIL import Image
from torch.autograd import Variable
from albumentations.pytorch.functional import img_to_tensor

def create_dir(directory):
    """Create directory if it does not exist."""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

@torch.no_grad()
def predict_from_image(model, path_images, path_out, threshold=0.5):
    """Predict segmentation masks from input images using the provided model."""
    create_dir(path_out)
    for path_image in glob(os.path.join(path_images, "*.png")) + glob(os.path.join(path_images, "*.jpg")):
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_tensor(image)
        image = Variable(image.unsqueeze(0).float())

        prediction = model(image)
        prediction = (prediction > threshold).float() * 255
        prediction = prediction.squeeze(0).numpy().astype(np.uint8)

        output_path = os.path.join(path_out, os.path.basename(path_image))
        cv2.imwrite(output_path, prediction)
