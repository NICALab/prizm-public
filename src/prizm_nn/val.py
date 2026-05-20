import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pandas as pd
import segmentation_models_pytorch as smp

from prizm_nn.dataset import PRIZM_Dataset
from model.model import DeepLabV3, DeepLabV3Plus
from prizm_nn.utils import *
import numpy as np
from torchvision.utils import save_image

def compute_metrics(tp, fp, fn, tn, classes, filename=None):
    metrics_row = {}

    # Add filename
    if filename:
        metrics_row["Filename"] = filename
    else:
        metrics_row["Filename"] = "Overall"

    # Compute metrics for each class
    for i, class_name in enumerate(classes):
        metrics_row[f"Accuracy ({class_name})"] = smp.metrics.accuracy(tp[:, i:i+1], fp[:, i:i+1], fn[:, i:i+1], tn[:, i:i+1], reduction="micro").item()
        metrics_row[f"IoU ({class_name})"] = smp.metrics.iou_score(tp[:, i:i+1], fp[:, i:i+1], fn[:, i:i+1], tn[:, i:i+1], reduction="micro").item()
        metrics_row[f"Precision ({class_name})"] = smp.metrics.precision(tp[:, i:i+1], fp[:, i:i+1], fn[:, i:i+1], tn[:, i:i+1], reduction="micro").item()
        metrics_row[f"Dice ({class_name})"] = smp.metrics.f1_score(tp[:, i:i+1], fp[:, i:i+1], fn[:, i:i+1], tn[:, i:i+1], reduction="micro").item()

    # Compute metrics for all classes combined
    metrics_row["Accuracy (Overall)"] = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
    metrics_row["IoU (Overall)"] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    metrics_row["Precision (Overall)"] = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
    metrics_row["Dice (Overall)"] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

    # Make into a row
    metrics_row_df = pd.DataFrame([metrics_row])

    return metrics_row_df

if __name__ == "__main__":
    args = parse_args()

    # Create directories for results
    experiment_path = os.path.join(args.results_dir, args.exp_name)
    log_dir = os.path.join(experiment_path, "log")
    model_dir = os.path.join(experiment_path, "models")
    img_dir = os.path.join(experiment_path, "images", f"{args.test_epoch}", "val")
    os.makedirs(img_dir, exist_ok=True)
    analysis_dir = os.path.join(experiment_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

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

    # Load dataset
    dataset = PRIZM_Dataset(args, mode="val")

    # Split dataset
    _, test_data = train_test_split(dataset, test_size=args.test_split, random_state=args.seed)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    print(f"Test samples: {len(test_data)}")

    # Validation loop
    model.eval()
    all_tp, all_fp, all_fn, all_tn = [], [], [], []
    classes = ["Background", "Ventricle", "Atrium"]
    metrics_df = pd.DataFrame()

    with torch.no_grad():
        for i, (image, mask, image_path) in enumerate(tqdm(test_loader, desc="Validation", dynamic_ncols=True)):
            image, mask= image.to(device), mask.to(device)
            outputs = model(image)
            print(image_path)
            # print(f"Preds shape: {outputs.shape} | Mask shape: {mask.shape}")
            # print(f"Preds range: {outputs.min()} - {outputs.max()} | Mask range: {mask.min()} - {mask.max()}")

            preds = torch.argmax(outputs, dim=1)
            # target = torch.argmax(mask, dim=1)

            # Convert to one-hot encoding
            preds = F.one_hot(preds, num_classes=args.num_classes).permute(0, 3, 1, 2)
            # target_one_hot = F.one_hot(target, num_classes=args.num_classes).permute(0, 3, 1, 2)

            # print(f"Preds shape: {preds.shape} | Mask shape: {mask.shape}")
            # print(f"Preds range: {preds.min()} - {preds.max()} | Mask range: {mask.min()} - {mask.max()}")
            # exit()
            # Compute stats for metrics
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, mask, mode='multilabel', threshold=0.5
            )

            filename = os.path.basename(image_path[0])
            metrics_df_i = compute_metrics(tp, fp, fn, tn, classes, filename=filename)
            metrics_df = pd.concat([metrics_df, metrics_df_i], ignore_index=True)

            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

            # Save the segmentation result
            pred_argmax = torch.argmax(outputs, dim=1)
            pred_argmax = torch.nn.functional.one_hot(pred_argmax, num_classes=3)
            pred_argmax = pred_argmax.permute(0, 3, 1, 2).float()

            image = F.interpolate(image, size=(300, 300), mode='bilinear')
            pred_argmax = F.interpolate(pred_argmax, size=(300, 300), mode='nearest')
            mask = F.interpolate(mask.float(), size=(300, 300), mode='nearest')

            pred_argmax = torch.argmax(pred_argmax, dim=1)
            mask = torch.argmax(mask, dim=1)

            mapping = torch.tensor([0, 2, 4]).to(device)
            pred_argmax = mapping[pred_argmax]
            mask = mapping[mask]

            pred_argmax = pred_argmax.squeeze().cpu().numpy().astype('uint8')
            mask = mask.squeeze().cpu().numpy().astype('uint8')

            pred_argmax = Image.fromarray(pred_argmax)
            mask = Image.fromarray(mask)

            save_image(image.squeeze(), os.path.join(img_dir, f"image_{i}.png"))
            pred_argmax.save(os.path.join(img_dir, f"pred_{i}.png"))
            mask.save(os.path.join(img_dir, f"gt_{i}.png"))

    # Aggregate statistics across batches
    tp = torch.cat(all_tp, dim=0).detach().cpu()
    fp = torch.cat(all_fp, dim=0).detach().cpu()
    fn = torch.cat(all_fn, dim=0).detach().cpu()
    tn = torch.cat(all_tn, dim=0).detach().cpu()  

    metrics_overall_df = compute_metrics(tp, fp, fn, tn, classes)

    # Append overall metrics to the top of the dataframe
    metrics_df = pd.concat([metrics_overall_df, metrics_df], ignore_index=True)

    # Save metrics to Excel
    excel_path = os.path.join(analysis_dir, "validation_metrics.xlsx")

    # Save to Excel
    metrics_df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")
