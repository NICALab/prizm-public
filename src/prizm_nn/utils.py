import argparse
import torch


def stretchlim(image, tol=(0.01, 0.99)):
    """
    Compute the stretch limits for contrast adjustment.
    """
    flattened = image.flatten()
    sorted_vals, _ = torch.sort(flattened)
    n_pixels = sorted_vals.shape[0]

    # Get the lower and upper limits based on the tolerance
    low_idx = int(tol[0] * n_pixels)
    high_idx = int(tol[1] * n_pixels) - 1

    low = sorted_vals[low_idx]
    high = sorted_vals[high_idx]

    # If the image is flat, avoid invalid range
    if low == high:
        return torch.tensor([0.0, 1.0], device=image.device)

    return torch.tensor([low, high], device=image.device)

def imadjust(image, in_range=None, out_range=(0, 1)):
    """
    Adjust the intensity values of an image.
    """
    if in_range is None:
        in_range = (image.min(), image.max())

    # Clip image values to in_range
    image = torch.clamp(image, min=in_range[0], max=in_range[1])

    # Normalize to [0, 1] and scale to out_range
    normalized = (image - in_range[0]) / (in_range[1] - in_range[0])
    scaled = normalized * (out_range[1] - out_range[0]) + out_range[0]

    return scaled

def overlay_segmentation(image, pred_argmax_channel):
    # Ensure image and pred_argmax_channel are on the same device and correct dtype
    image = image.to(pred_argmax_channel.device).float()

    # Convert image to three channels (RGB-like) for visualization.
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [N,C,H,W], got shape={tuple(image.shape)}")
    if image.shape[1] == 1:
        rgb_image = image.repeat(1, 3, 1, 1)
    elif image.shape[1] >= 3:
        rgb_image = image[:, :3, :, :]
    else:
        # Fallback for unusual channel counts.
        rgb_image = image.repeat(1, 3, 1, 1)[:, :3, :, :]
    
    # Create color masks for the segmentation maps
    # Using red color for channel 1 and green color for channel 2
    red_mask = pred_argmax_channel[:, 1:2, :, :]  # Channel 1 mask
    green_mask = pred_argmax_channel[:, 2:3, :, :]  # Channel 2 mask
    
    # Apply the masks with specific colors
    # Adjust the intensity if required, e.g., keeping the image background intensity lower
    overlay = rgb_image.clone()
    overlay[:, 0:1, :, :] = (overlay[:, 0:1, :, :] * (1 - red_mask)) + red_mask
    overlay[:, 1:2, :, :] = (overlay[:, 1:2, :, :] * (1 - green_mask)) + green_mask
    
    return torch.clamp(overlay, 0.0, 1.0)

def total_variation_loss(y):
    # Calculate the differences between adjacent pixels
    diff_i = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    diff_j = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    
    # Sum up the differences and normalize by the number of elements
    tv_loss = torch.sum(diff_i) + torch.sum(diff_j)
    num_elements = (y.size(2) - 1) * y.size(3) + (y.size(3) - 1) * y.size(2)
    normalized_tv_loss = tv_loss / num_elements
    return normalized_tv_loss

def squared_difference_loss(y):
    # Calculate the squared differences between adjacent pixels
    diff_i = (y[:, :, 1:, :] - y[:, :, :-1, :]) ** 2
    diff_j = (y[:, :, :, 1:] - y[:, :, :, :-1]) ** 2
    
    # Sum up the squared differences and normalize by the number of elements
    sd_loss = torch.sum(diff_i) + torch.sum(diff_j)
    num_elements = (y.size(2) - 1) * y.size(3) + (y.size(3) - 1) * y.size(2)
    normalized_sd_loss = sd_loss / num_elements
    return normalized_sd_loss

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepLabV3+ model on medical images.")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for dataset splitting.')
    
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training and validation.')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing.')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Batch size for held-out test evaluation.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine', 'none'], help='Learning rate scheduler to use.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lr_drop_factor', type=float, default=0.3, help='Factor to reduce learning rate by.')
    parser.add_argument('--lr_drop_period', type=int, default=5, help='Period in epochs to reduce learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer (MATLAB SGDM default L2 regularization is 1e-4).')
    parser.add_argument('--tversky_alpha', type=float, default=0.01, help='Alpha parameter for Tversky loss.')
    parser.add_argument('--tversky_beta', type=float, default=0.99, help='Beta parameter for Tversky loss.')
    parser.add_argument('--train_split', type=float, default=0.90, help='Proportion of paired data used for training.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of paired data used for validation.')
    parser.add_argument('--test_split', type=float, default=0.05, help='Proportion of paired data used for held-out testing.')
    parser.add_argument('--dataset_dir', type=str, default="/media/HDD1/josh/prizm/data", help='Path to the dataset directory.')
    parser.add_argument('--test_dataset_dir', type=str, default="/media/HDD1/josh/prizm/data/20240827/preprocessing", help='Path to the test dataset directory.')
    parser.add_argument('--augmentation', type=int, default=0, help='Whether to apply data augmentation during training.')
    parser.add_argument('--random_rot90', type=int, default=0, help='Apply random 0/90/180/270 degree joint rotation to training image/mask pairs.')
    parser.add_argument('--exp_name', type=str, default="11242024_init", help='Experiment name for logging and results.')
    parser.add_argument('--results_dir', type=str, default="/media/HDD1/josh/prizm/results/11242024", help='Base directory for results.')
    
    parser.add_argument('--log_interval', type=int, default=50, help='Iteration interval for training logs (MATLAB VerboseFrequency default used in the live script is 50).')
    parser.add_argument('--val_interval', type=int, default=1, help='[Deprecated] Unused legacy arg.')
    parser.add_argument('--save_model_interval', type=int, default=1, help='Interval (in epochs) to save the model.')
    parser.add_argument(
        '--validation_interval',
        '--validation_frequency',
        dest='validation_interval',
        type=int,
        default=50,
        help='Iteration interval for validation checks (MATLAB ValidationFrequency default is 50 when unspecified).',
    )
    parser.add_argument('--validation_patience', type=int, default=30, help='Early-stop patience in validation checks.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0, help='Minimum improvement in the chosen early-stop metric to reset patience.')
    parser.add_argument(
        '--early_stopping_metric',
        type=str,
        default='val_loss',
        choices=['val_loss', 'val_dice'],
        help='Metric used for best-checkpoint selection and early stopping.',
    )
    parser.add_argument('--save_best_model', type=int, default=1, help='Save best model checkpoint by the chosen early-stop metric.')
    
    parser.add_argument('--model', type=str, default='deeplabv3plus', choices=['deeplabv3', 'deeplabv3plus'], help='Model architecture to use.')
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        help='Backbone architecture for the model.',
    )
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes for segmentation.')
    parser.add_argument('--encoder_depth', type=int, default=5, help='Depth of the encoder in the model.')
    parser.add_argument('--decoder_channels', type=int, default=256, help='Number of channels in the decoder.')
    parser.add_argument('--encoder_output_stride', type=int, default=16, help='Output stride of the encoder.')
    parser.add_argument('--decoder_atrous_rates', type=int, nargs='+', default=[6, 12, 18], help='Atrous rates for the decoder in DeepLabV3+.')
    parser.add_argument('--input_channels', type=int, default=3, help='Model input channels (MATLAB parity uses 3).')
    parser.add_argument('--encoder_weights', type=str, default='none', help='Encoder weight initialization. MATLAB deeplabv3plusLayers with resnet50 uses a pretrained backbone.')
    
    parser.add_argument('--include_background', type=int, default=0, help='Include background class in the loss calculation.')
    parser.add_argument('--smooth_loss', type=str, default='tv', choices=['tv', 'sd'], help='Smoothness loss to use.')
    parser.add_argument('--lambda_smooth', type=float, default=0.0, help='Weight for total variation loss.')
    parser.add_argument('--matlab_parity', type=int, default=1, help='Enable MATLAB-parity data split and preprocessing flow.')
    parser.add_argument('--apply_random_transform_to_eval', type=int, default=0, help='Apply MATLAB-like random train augmentation on val/test too. Default 0 keeps validation/test deterministic.')
    parser.add_argument(
        '--monitor_matlab_results_root',
        type=str,
        default='',
        help='Optional MATLAB result root used for deterministic monitored-session evaluation during training.',
    )
    parser.add_argument(
        '--monitor_sessions',
        type=str,
        nargs='*',
        default=[],
        help='Session specs like ISO_50|Series048 to monitor against MATLAB masks during training.',
    )
    parser.add_argument(
        '--monitor_batch_size',
        type=int,
        default=8,
        help='Batch size for monitored MATLAB-session evaluation.',
    )
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers.')
    parser.add_argument('--pin_memory', type=int, default=1, help='Dataloader pin_memory.')
    parser.add_argument(
        '--preload_to_memory',
        type=int,
        default=1,
        help='Preload the active training/eval datasets into RAM before training starts.',
    )

    parser.add_argument('--test_epoch', type=int, default=100, help='Test epoch')
    return parser.parse_args()
