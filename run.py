import argparse
import cv2
import torch

from model.unet import UNet
from utils import predict_from_image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run U-Net model for aerial imagery detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained U-Net model")
    parser.add_argument("--input_image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output segmentation mask")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for inference (cpu or cuda)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load model
    try:
        device = torch.device(args.device)
        model = UNet().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device)["model"].state_dict())
        model.eval()
    except Exception as e:
        print("Error loading the model:", e)
        return

    # Perform inference
    try:
        predict_from_image(model, args.input_image_path, args.output_dir, device=device)
        print("Segmentation mask saved successfully.")
    except Exception as e:
        print("Error during inference:", e)

if __name__ == "__main__":
    main()
