import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from monai.losses import TverskyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from prizm_nn.dataset import PRIZM_Dataset
from prizm_nn.model import DeepLabV3, DeepLabV3Plus
import torch.nn.functional as F
from prizm_nn.utils import *
import torch

import glob
import re
import imageio

def extract_file_info(template_string):
    # Define a regex pattern to extract the components
    pattern = r"cropped_(.+)_t(\d+)_ch\d+\.jpg"
    
    # Match the pattern
    match = re.search(pattern, template_string)
    
    if match:
        # Extract the components
        file_name = match.group(1)
        frame_number = match.group(2)
        full_file_name = f"{file_name}_t{frame_number}"
        
        # Return the results as a tuple
        return full_file_name, file_name, frame_number
    else:
        raise ValueError(f"The input string does not match the expected pattern: {template_string}")
    
if __name__ == "__main__":
    args = parse_args()

    # Create directories for results
    experiment_path = os.path.join(args.results_dir, args.exp_name)
    log_dir = os.path.join(experiment_path, "log")
    model_dir = os.path.join(experiment_path, "models")
    img_dir = os.path.join(experiment_path, "images", f"{args.test_epoch}", "test")
    frame_dir = os.path.join(img_dir, "frame")
    gif_dir = os.path.join(img_dir, "gif")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'deeplabv3':
        model = DeepLabV3(num_classes=args.num_classes, in_channels=args.input_channels).to(device)
    elif args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(
            num_classes=args.num_classes,
            decoder_atrous_rates=args.decoder_atrous_rates,
            backbone=args.backbone,
            encoder_depth=args.encoder_depth,
            decoder_channels=args.decoder_channels,
            encoder_output_stride=args.encoder_output_stride,
            in_channels=args.input_channels,
        ).to(device)

    # Load model
    model_path = os.path.join(model_dir, f"model_epoch_{args.test_epoch}.pth")
    model.load_state_dict(torch.load(model_path))
    if args.model == 'deeplabv3':
        transform_img = transforms.Compose([
            transforms.Resize((300, 300)),  # Resize image to 300x300
            transforms.Grayscale(num_output_channels=1),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
            transforms.ToTensor()  # Convert image to a tensor
        ])
    elif args.model == 'deeplabv3plus':
        transform_img = transforms.Compose([
            transforms.Resize((304, 304)),  # Resize image to 300x300
            transforms.Grayscale(num_output_channels=1),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
            transforms.ToTensor()  # Convert image to a tensor
        ])
        
    test_files = glob.glob(os.path.join(args.test_dataset_dir, "*"))
    
    video_frames = {}
    
    # Testing loop
    model.eval()
    with torch.no_grad():
        for i, test_file in enumerate(tqdm(test_files, desc="Testing", dynamic_ncols=True)):
            full_file_name, file_name, frame_number = extract_file_info(test_file)
            
            if file_name not in video_frames:
                video_frames[file_name] = {"image": [], "seg_pred": [], "overlay_pred": []}
            # import pdb; pdb.set_trace()
            image = Image.open(test_file)
            image = transform_img(image)
            
            image = image.unsqueeze(0).to(device)
            
            J = stretchlim(image[0, 0])
            J_modified = J.clone()
            J_modified[0] = J[0] * 1.0
            J_modified[1] = J[1] * 0.90
            image = imadjust(image, in_range=J_modified)
            
            outputs = model(image)

            # Save the segmentation result
            pred_argmax = torch.argmax(outputs, dim=1)  # Add channel dimension
            pred_argmax = torch.nn.functional.one_hot(pred_argmax, num_classes=3)  # (B, H, W, 3)
            pred_argmax = pred_argmax.permute(0, 3, 1, 2).float()  # Reshape to (B, 3, H, W)

            # Resize the image, pred, mask to 300 x 300
            image = F.interpolate(image, size=(300, 300), mode='bilinear')
            pred_argmax = F.interpolate(pred_argmax, size=(300, 300), mode='nearest')
            pred_argmax_channel = pred_argmax.clone()

            pred_argmax = torch.argmax(pred_argmax, dim=1)

            mapping = torch.tensor([0, 2, 4]).to(device)

            pred_argmax = mapping[pred_argmax]
            pred_argmax = pred_argmax.squeeze().cpu().numpy().astype('uint8')
            pred_argmax = Image.fromarray(pred_argmax)
            
            overlay = overlay_segmentation(image, pred_argmax_channel)

            save_image(image.squeeze(), os.path.join(frame_dir, f"{full_file_name}_image.png"))
            # save_image(overlay.squeeze(), os.path.join(frame_dir, f"{full_file_name}_overlay.png"))
            # save_image(pred_argmax_channel.squeeze(), os.path.join(frame_dir, f"{full_file_name}_seg_pred.png"))
            pred_argmax.save(os.path.join(frame_dir, f"{full_file_name}_segmentation.png"))

            # Append to video_frames
            video_frames[file_name]["image"].append(image.cpu().numpy().squeeze(0).transpose(1, 2, 0))
            video_frames[file_name]["seg_pred"].append(pred_argmax_channel.cpu().numpy().squeeze(0).transpose(1, 2, 0))
            video_frames[file_name]["overlay_pred"].append(overlay.cpu().numpy().squeeze(0).transpose(1, 2, 0))
    
    # Create GIFs
    for file_name, frames in video_frames.items():
        image_frames = [Image.fromarray((img.squeeze() * 255).astype(np.uint8)) for img in frames["image"]]
        seg_pred_frames = [Image.fromarray((img * 255).astype(np.uint8)) for img in frames["seg_pred"]]
        overlay_pred_frames = [Image.fromarray((img * 255).astype(np.uint8)) for img in frames["overlay_pred"]]
        
        # Save original image frames as a GIF
        imageio.mimsave(os.path.join(gif_dir, f"{file_name}_images.gif"), image_frames, fps=5)
        
        # Save segmentation predictions as a GIF
        imageio.mimsave(os.path.join(gif_dir, f"{file_name}_segmentation.gif"), seg_pred_frames, fps=5)
        
        # Save overlay predictions as a GIF
        imageio.mimsave(os.path.join(gif_dir, f"{file_name}_overlay.gif"), overlay_pred_frames, fps=5)
