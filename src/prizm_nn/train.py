from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from prizm_nn.dataset import (
    MatlabMonitorSegDataset,
    MatlabParitySegDataset,
    discover_matlab_monitor_pairs,
    PRIZM_Dataset,
    discover_matlab_style_pairs,
    split_matlab_style_pairs,
)
from prizm_nn.utils import (
    overlay_segmentation,
    parse_args,
    squared_difference_loss,
    total_variation_loss,
)
from model.model import DeepLabV3, DeepLabV3Plus

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
    _TB_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False
    _TB_IMPORT_ERROR = e

try:
    from monai.losses import TverskyLoss as MonaiTverskyLoss
except Exception:  # pragma: no cover
    MonaiTverskyLoss = None


CLASS_NAMES = ["Background", "Ventricle", "Atrium"]


class MatlabTverskyLoss(torch.nn.Module):
    """Exact reduction semantics for the MATLAB tverskyPixelClassificationLayer."""

    def __init__(self, alpha: float = 0.01, beta: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target = target_onehot.float()

        # MATLAB layer computes per-class, per-sample TP/FP/FN by summing over H,W
        # on arrays shaped [H,W,C,N], then sums over classes and averages over N.
        # PyTorch tensors here are [N,C,H,W], so sum over spatial dims only.
        dims = (2, 3)
        tp = torch.sum(probs * target, dim=dims)
        fp = torch.sum(probs * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - probs) * target, dim=dims)

        loss_tic = 1.0 - ((tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps))
        loss_ti = torch.sum(loss_tic, dim=1)
        return torch.mean(loss_ti)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(args, device: torch.device):
    if args.model == "deeplabv3":
        model = DeepLabV3(num_classes=args.num_classes, in_channels=args.input_channels)
    else:
        encoder_weights = args.encoder_weights
        if isinstance(encoder_weights, str) and encoder_weights.lower() in {"", "none", "null"}:
            encoder_weights = None
        model = DeepLabV3Plus(
            num_classes=args.num_classes,
            decoder_atrous_rates=args.decoder_atrous_rates,
            backbone=args.backbone,
            encoder_depth=args.encoder_depth,
            decoder_channels=args.decoder_channels,
            encoder_output_stride=args.encoder_output_stride,
            in_channels=args.input_channels,
            encoder_weights=encoder_weights,
        )
    return model.to(device)


def _build_dataloaders(args):
    if bool(args.matlab_parity):
        pairs = discover_matlab_style_pairs(args.dataset_dir)
        if len(pairs) == 0:
            raise ValueError(
                f"No MATLAB-style image/mask pairs found in {args.dataset_dir}. "
                "Expected paired image + '*_Simple Segmentation*' mask files."
            )

        split = split_matlab_style_pairs(
            pairs=pairs,
            train_ratio=float(args.train_split),
            val_ratio=float(args.val_split),
            test_ratio=float(args.test_split),
            seed=int(args.seed),
        )

        train_dataset = MatlabParitySegDataset(
            split["train"],
            apply_random_transform=True,
            random_rot90=bool(args.random_rot90),
            preload_to_memory=bool(args.preload_to_memory),
        )
        eval_random = bool(args.apply_random_transform_to_eval)
        val_dataset = MatlabParitySegDataset(
            split["val"],
            apply_random_transform=eval_random,
            random_rot90=False,
            preload_to_memory=bool(args.preload_to_memory),
        )
        test_dataset = MatlabParitySegDataset(
            split["test"],
            apply_random_transform=eval_random,
            random_rot90=False,
            preload_to_memory=bool(args.preload_to_memory),
        )
    else:
        # Backward-compatible legacy path.
        dataset = PRIZM_Dataset(args, mode="train")
        train_data, holdout = train_test_split(
            dataset,
            test_size=(args.val_split + args.test_split),
            random_state=args.seed,
        )
        holdout_rel_test = args.test_split / (args.val_split + args.test_split)
        val_data, test_data = train_test_split(
            holdout,
            test_size=holdout_rel_test,
            random_state=args.seed,
        )
        train_dataset = train_data
        val_dataset = val_data
        test_dataset = test_data

    pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def _build_monitor_loaders(args):
    if not args.monitor_matlab_results_root or not args.monitor_sessions:
        return {}

    monitor_pairs = discover_matlab_monitor_pairs(
        matlab_results_root=args.monitor_matlab_results_root,
        session_specs=args.monitor_sessions,
    )
    pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    monitor_loaders = {}
    for session_name, pairs in monitor_pairs.items():
        dataset = MatlabMonitorSegDataset(
            pairs,
            preload_to_memory=bool(args.preload_to_memory),
        )
        monitor_loaders[session_name] = DataLoader(
            dataset,
            batch_size=max(int(args.monitor_batch_size), 1),
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    return monitor_loaders


def _batch_pixel_accuracy(logits: torch.Tensor, target_onehot: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    target = torch.argmax(target_onehot, dim=1)
    acc = (pred == target).float().mean().item()
    return float(acc * 100.0)


def _update_confusion(conf: np.ndarray, pred_idx: torch.Tensor, target_idx: torch.Tensor) -> None:
    n_cls = conf.shape[0]
    p = pred_idx.reshape(-1).detach().cpu().numpy().astype(np.int64)
    t = target_idx.reshape(-1).detach().cpu().numpy().astype(np.int64)
    valid = (t >= 0) & (t < n_cls) & (p >= 0) & (p < n_cls)
    if not np.any(valid):
        return
    hist = np.bincount(
        n_cls * t[valid] + p[valid],
        minlength=n_cls * n_cls,
    ).reshape(n_cls, n_cls)
    conf += hist


def _metrics_from_confusion(conf: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps = 1e-12
    n_cls = conf.shape[0]
    total = float(conf.sum())

    tp_all = float(np.trace(conf))
    fp_all = float(conf.sum(axis=0).sum() - tp_all)
    fn_all = float(conf.sum(axis=1).sum() - tp_all)
    tn_all = float(total - tp_all - fp_all - fn_all)

    class_metrics = []
    for i in range(n_cls):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        tn = float(total - tp - fp - fn)

        acc = (tp + tn) / (total + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        class_metrics.append(
            {
                "ClassName": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class{i}",
                "Accuracy": acc,
                "IoU": iou,
                "Precision": precision,
                "Dice": dice,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )

    overall = pd.DataFrame(
        [
            {
                "Accuracy (Overall)": (tp_all + tn_all) / (total + eps),
                "IoU (Overall)": tp_all / (tp_all + fp_all + fn_all + eps),
                "Precision (Overall)": tp_all / (tp_all + fp_all + eps),
                "Dice (Overall)": (2.0 * tp_all) / (2.0 * tp_all + fp_all + fn_all + eps),
                "TotalPixels": total,
            }
        ]
    )
    class_df = pd.DataFrame(class_metrics)
    conf_df = pd.DataFrame(conf, columns=CLASS_NAMES[:n_cls], index=CLASS_NAMES[:n_cls])
    return overall, class_df, conf_df


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    return_preview: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_pixels = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    preview: Optional[Dict[str, torch.Tensor]] = None

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(image)
            loss = criterion(logits, mask)
            total_loss += float(loss.item())
            total_batches += 1

            pred_idx = torch.argmax(logits, dim=1)
            target_idx = torch.argmax(mask, dim=1)
            total_correct += int((pred_idx == target_idx).sum().item())
            total_pixels += int(pred_idx.numel())
            _update_confusion(conf, pred_idx, target_idx)

            if return_preview and preview is None:
                pred_onehot = (
                    torch.nn.functional.one_hot(pred_idx, num_classes=num_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                max_n = min(4, image.shape[0])
                preview = {
                    "image": image[:max_n].detach().cpu(),
                    "mask": mask[:max_n].detach().cpu(),
                    "pred_onehot": pred_onehot[:max_n].detach().cpu(),
                }

    avg_loss = total_loss / max(total_batches, 1)
    pixel_acc = 100.0 * total_correct / max(total_pixels, 1)
    overall_df, class_df, conf_df = _metrics_from_confusion(conf)
    return avg_loss, pixel_acc, overall_df, class_df, conf_df, preview


def _last_finite(values: np.ndarray, default: float = np.nan) -> float:
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(default)
    return float(finite[-1])


def _log_legacy_metric_scalars(
    scalar_logger,
    phase: str,
    loss_value: float,
    overall_df: pd.DataFrame,
    class_df: pd.DataFrame,
    step: int,
) -> None:
    """
    Restore old TensorBoard scalar names used by the original trainer.
    """
    if phase not in {"Train", "Validation", "Test"}:
        return

    if phase == "Train":
        scalar_logger("Train/00. Training Loss", float(loss_value), step)
    elif phase == "Validation":
        scalar_logger("Validation/00. Validation Loss", float(loss_value), step)
    else:
        scalar_logger("Test/00. Test Loss", float(loss_value), step)

    entries = []
    # Match old behavior for train/val: skip background class. For test, log all classes.
    class_names = ("Ventricle", "Atrium") if phase in {"Train", "Validation"} else tuple(
        class_df["ClassName"].tolist()
    )
    for cls_name in class_names:
        row = class_df[class_df["ClassName"] == cls_name]
        if row.empty:
            continue
        row = row.iloc[0]
        entries.extend(
            [
                (f"Accuracy ({cls_name})", float(row["Accuracy"])),
                (f"IoU ({cls_name})", float(row["IoU"])),
                (f"Precision ({cls_name})", float(row["Precision"])),
                (f"Dice ({cls_name})", float(row["Dice"])),
                (f"TP ({cls_name})", float(row["TP"])),
                (f"FP ({cls_name})", float(row["FP"])),
                (f"FN ({cls_name})", float(row["FN"])),
                (f"TN ({cls_name})", float(row["TN"])),
            ]
        )

    if not overall_df.empty:
        o = overall_df.iloc[0]
        entries.extend(
            [
                ("Accuracy (Overall)", float(o["Accuracy (Overall)"])),
                ("IoU (Overall)", float(o["IoU (Overall)"])),
                ("Precision (Overall)", float(o["Precision (Overall)"])),
                ("Dice (Overall)", float(o["Dice (Overall)"])),
                ("TotalPixels", float(o["TotalPixels"])),
            ]
        )

    prefix = "Train" if phase == "Train" else ("Validation" if phase == "Validation" else "Test")
    for i, (name, value) in enumerate(entries, start=1):
        tag = f"{prefix}/{i:02d}. {name}"
        scalar_logger(tag, value, step)


def _log_confusion_matrix_scalars(
    scalar_logger,
    phase: str,
    conf_df: pd.DataFrame,
    conf_norm_df: pd.DataFrame,
    step: int,
) -> None:
    prefix = "Train" if phase == "Train" else ("Validation" if phase == "Validation" else "Test")
    for row_name in conf_df.index:
        for col_name in conf_df.columns:
            raw_val = float(conf_df.loc[row_name, col_name])
            norm_val = float(conf_norm_df.loc[row_name, col_name])
            scalar_logger(f"{prefix}/ConfusionRaw/{row_name}_to_{col_name}", raw_val, step)
            scalar_logger(f"{prefix}/ConfusionNorm/{row_name}_to_{col_name}", norm_val, step)


def _to_three_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] >= 3:
        return x[:, :3, :, :]
    return x.repeat(1, 3, 1, 1)[:, :3, :, :]


def _mask_onehot_to_color_rgb(mask: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(mask, dim=1)
    rgb = torch.zeros((mask.shape[0], 3, mask.shape[2], mask.shape[3]), dtype=torch.float32)
    rgb[:, 0][idx == 1] = 1.0
    rgb[:, 1][idx == 2] = 1.0
    return rgb


def _mask_pair_mismatch_rgb(mask: torch.Tensor, pred_onehot: torch.Tensor) -> torch.Tensor:
    gt_idx = torch.argmax(mask, dim=1)
    pred_idx = torch.argmax(pred_onehot, dim=1)
    mismatch = gt_idx != pred_idx
    rgb = torch.zeros((mask.shape[0], 3, mask.shape[2], mask.shape[3]), dtype=torch.float32)

    gt_v = (gt_idx == 1) & mismatch
    gt_a = (gt_idx == 2) & mismatch
    pred_v = (pred_idx == 1) & mismatch
    pred_a = (pred_idx == 2) & mismatch

    rgb[:, 0][gt_v] = 1.0
    rgb[:, 1][gt_a] = 1.0
    rgb[:, 0][pred_v] = 1.0
    rgb[:, 2][pred_v] = 1.0
    rgb[:, 1][pred_a] = 1.0
    rgb[:, 2][pred_a] = 1.0
    return rgb


def _sanitize_tag_component(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _log_legacy_preview_images(
    writer: SummaryWriter,
    phase: str,
    image: torch.Tensor,
    mask: torch.Tensor,
    pred_onehot: torch.Tensor,
    step: int,
) -> None:
    """
    Restore old TensorBoard image tags:
    - {phase}/00. ... Overlay
    - {phase}/01. ... Input
    - {phase}/02. ... Ground Truth
    - {phase}/03. ... Prediction (Argmax)
    """
    max_n = min(4, image.shape[0])
    image = image[:max_n]
    mask = mask[:max_n]
    pred_onehot = pred_onehot[:max_n]

    overlay = overlay_segmentation(image, pred_onehot)
    input_rgb = _to_three_channel(image.float())
    gt_rgb = _to_three_channel(mask.float())
    pred_rgb = _to_three_channel(pred_onehot.float())

    if phase == "Train":
        phase_name, prefix = "Training", "Train"
    elif phase == "Validation":
        phase_name, prefix = "Validation", "Validation"
    else:
        phase_name, prefix = "Test", "Test"
    writer.add_images(f"{prefix}/00. {phase_name} Overlay", overlay, step)
    writer.add_images(f"{prefix}/01. {phase_name} Input", input_rgb, step)
    writer.add_images(f"{prefix}/02. {phase_name} Ground Truth", gt_rgb, step)
    writer.add_images(f"{prefix}/03. {phase_name} Prediction (Argmax)", pred_rgb, step)


def _log_monitor_metric_scalars(
    scalar_logger,
    tag_prefix: str,
    loss_value: float,
    pixel_accuracy: float,
    overall_df: pd.DataFrame,
    class_df: pd.DataFrame,
    step: int,
) -> None:
    scalar_logger(f"{tag_prefix}/Loss", float(loss_value), step)
    scalar_logger(f"{tag_prefix}/PixelAccuracy", float(pixel_accuracy), step)
    if not overall_df.empty:
        row = overall_df.iloc[0]
        scalar_logger(f"{tag_prefix}/Overall/Accuracy", float(row["Accuracy (Overall)"]), step)
        scalar_logger(f"{tag_prefix}/Overall/IoU", float(row["IoU (Overall)"]), step)
        scalar_logger(f"{tag_prefix}/Overall/Precision", float(row["Precision (Overall)"]), step)
        scalar_logger(f"{tag_prefix}/Overall/Dice", float(row["Dice (Overall)"]), step)
    for _, row in class_df.iterrows():
        class_tag = _sanitize_tag_component(str(row["ClassName"]))
        scalar_logger(f"{tag_prefix}/{class_tag}/Accuracy", float(row["Accuracy"]), step)
        scalar_logger(f"{tag_prefix}/{class_tag}/IoU", float(row["IoU"]), step)
        scalar_logger(f"{tag_prefix}/{class_tag}/Precision", float(row["Precision"]), step)
        scalar_logger(f"{tag_prefix}/{class_tag}/Dice", float(row["Dice"]), step)


def _log_monitor_preview_images(
    writer: SummaryWriter,
    phase_prefix: str,
    session_name: str,
    image: torch.Tensor,
    mask: torch.Tensor,
    pred_onehot: torch.Tensor,
    step: int,
) -> None:
    max_n = min(4, image.shape[0])
    image = image[:max_n].float()
    mask = mask[:max_n].float()
    pred_onehot = pred_onehot[:max_n].float()

    session_tag = _sanitize_tag_component(session_name)
    input_rgb = _to_three_channel(image)
    gt_rgb = _mask_onehot_to_color_rgb(mask)
    pred_rgb = _mask_onehot_to_color_rgb(pred_onehot)
    overlay = overlay_segmentation(image, pred_onehot)
    mismatch_rgb = _mask_pair_mismatch_rgb(mask, pred_onehot)

    writer.add_images(f"{phase_prefix}/{session_tag}/00. Overlay", overlay, step)
    writer.add_images(f"{phase_prefix}/{session_tag}/01. Input", input_rgb, step)
    writer.add_images(f"{phase_prefix}/{session_tag}/02. MATLAB Ground Truth", gt_rgb, step)
    writer.add_images(f"{phase_prefix}/{session_tag}/03. Prediction", pred_rgb, step)
    writer.add_images(f"{phase_prefix}/{session_tag}/04. Mismatch", mismatch_rgb, step)


def main():
    args = parse_args()

    if bool(args.matlab_parity):
        matlab_live_script_expectations = {
            "apply_random_transform_to_eval": 1,
            "epochs": 100,
            "validation_patience": 30,
            "train_batch_size": 8,
            "validation_interval": 50,
            "log_interval": 50,
            "optimizer": "sgd",
            "lr_scheduler": "step",
            "lr_drop_factor": 0.3,
            "lr_drop_period": 5,
            "encoder_weights": "imagenet",
        }
        mismatches = []
        for key, expected in matlab_live_script_expectations.items():
            actual = getattr(args, key)
            if actual != expected:
                mismatches.append(f"{key}={actual!r} (MATLAB script uses {expected!r})")
        if mismatches:
            print("[WARN] matlab_parity=1 but CLI args diverge from the MATLAB training live script:")
            for item in mismatches:
                print(f"  - {item}")

    if not _TB_AVAILABLE:
        raise RuntimeError(
            "TensorBoard is required for training logs but is not available. "
            "Install it with: pip install tensorboard\n"
            f"Original import error: {_TB_IMPORT_ERROR!r}"
        )

    experiment_path = Path(args.results_dir) / args.exp_name
    log_dir = experiment_path / "log"
    model_dir = experiment_path / "models"
    analysis_dir = experiment_path / "analysis"
    for p in (experiment_path, log_dir, model_dir, analysis_dir):
        p.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("Arguments", str(args))
    scalar_records = []
    monitor_records: List[Dict[str, float]] = []

    def log_scalar(tag: str, value: float, step: int) -> None:
        v = float(value)
        s = int(step)
        writer.add_scalar(tag, v, s)
        scalar_records.append({"Step": s, "Tag": tag, "Value": v})

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(args, device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=args.lr_drop_period,
            gamma=args.lr_drop_factor,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=0.0
        )

    if bool(args.matlab_parity):
        criterion = MatlabTverskyLoss(
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
        )
    elif MonaiTverskyLoss is not None:
        criterion = MonaiTverskyLoss(
            include_background=bool(args.include_background),
            to_onehot_y=False,
            sigmoid=False,
            softmax=True,
            smooth_nr=1e-08,
            smooth_dr=1e-08,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
        )
    else:
        criterion = MatlabTverskyLoss(
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
        )

    train_loader, val_loader, test_loader = _build_dataloaders(args)
    monitor_loaders = _build_monitor_loaders(args)
    print(
        f"Training samples: {len(train_loader.dataset)} | "
        f"Validation samples: {len(val_loader.dataset)} | "
        f"Test samples: {len(test_loader.dataset)}"
    )
    if monitor_loaders:
        for session_name, loader in monitor_loaders.items():
            print(f"Monitor session: {session_name} | Frames: {len(loader.dataset)}")

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    lr_hist = []

    global_iter = 0
    best_val_loss = np.inf
    best_val_dice = -np.inf
    best_early_stop_score = np.inf if args.early_stopping_metric == "val_loss" else -np.inf
    bad_val_checks = 0
    stop_training = False
    best_model_path = model_dir / "model_best.pth"
    train_log_interval = max(int(args.log_interval), 1)
    validation_interval = max(int(args.validation_interval), 1)
    save_model_interval = max(int(args.save_model_interval), 1)

    def run_monitor_evaluations(phase_prefix: str, step: int) -> None:
        if not monitor_loaders:
            return
        for session_name, loader in monitor_loaders.items():
            m_loss, m_acc, m_overall, m_class, _, m_preview = evaluate_loader(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                return_preview=True,
            )
            tag_prefix = f"{phase_prefix}/{_sanitize_tag_component(session_name)}"
            _log_monitor_metric_scalars(
                scalar_logger=log_scalar,
                tag_prefix=tag_prefix,
                loss_value=m_loss,
                pixel_accuracy=m_acc,
                overall_df=m_overall,
                class_df=m_class,
                step=step,
            )
            if m_preview is not None:
                _log_monitor_preview_images(
                    writer=writer,
                    phase_prefix=phase_prefix,
                    session_name=session_name,
                    image=m_preview["image"],
                    mask=m_preview["mask"],
                    pred_onehot=m_preview["pred_onehot"],
                    step=step,
                )

            overall_row = m_overall.iloc[0] if not m_overall.empty else {}
            class_map = {
                str(row["ClassName"]): row
                for _, row in m_class.iterrows()
            }
            vent = class_map.get("Ventricle", {})
            atr = class_map.get("Atrium", {})
            monitor_records.append(
                {
                    "Step": float(step),
                    "Phase": phase_prefix,
                    "Session": session_name,
                    "Loss": float(m_loss),
                    "PixelAccuracy": float(m_acc),
                    "OverallDice": float(overall_row.get("Dice (Overall)", np.nan)),
                    "OverallIoU": float(overall_row.get("IoU (Overall)", np.nan)),
                    "VentricleDice": float(vent.get("Dice", np.nan)),
                    "VentricleIoU": float(vent.get("IoU", np.nan)),
                    "AtriumDice": float(atr.get("Dice", np.nan)),
                    "AtriumIoU": float(atr.get("IoU", np.nan)),
                }
            )

    for epoch in tqdm(range(args.epochs), dynamic_ncols=True, desc="Epochs"):
        model.train()
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_last_preview: Optional[Dict[str, torch.Tensor]] = None
        log_conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        log_loss_sum = 0.0
        log_batches = 0
        log_last_preview: Optional[Dict[str, torch.Tensor]] = None

        for image, mask in tqdm(train_loader, desc="Training", dynamic_ncols=True, leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, mask)

            if args.lambda_smooth > 0:
                if args.smooth_loss == "tv":
                    smooth_loss = total_variation_loss(logits)
                else:
                    smooth_loss = squared_difference_loss(logits)
                loss = loss + args.lambda_smooth * smooth_loss

            loss.backward()
            optimizer.step()

            global_iter += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            batch_acc = _batch_pixel_accuracy(logits, mask)
            pred_idx = torch.argmax(logits, dim=1)
            target_idx = torch.argmax(mask, dim=1)
            pred_onehot = (
                torch.nn.functional.one_hot(pred_idx, num_classes=args.num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )

            train_loss_hist.append(float(loss.item()))
            train_acc_hist.append(batch_acc)
            val_loss_hist.append(np.nan)
            val_acc_hist.append(np.nan)
            lr_hist.append(current_lr)

            _update_confusion(epoch_conf, pred_idx, target_idx)
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1
            epoch_last_preview = {
                "image": image.detach(),
                "mask": mask.detach(),
                "pred_onehot": pred_onehot.detach(),
            }
            _update_confusion(log_conf, pred_idx, target_idx)
            log_loss_sum += float(loss.item())
            log_batches += 1
            log_last_preview = {
                "image": image.detach(),
                "mask": mask.detach(),
                "pred_onehot": pred_onehot.detach(),
            }

            if global_iter % train_log_interval == 0:
                log_loss = log_loss_sum / max(log_batches, 1)
                log_overall, log_class, _ = _metrics_from_confusion(log_conf)
                log_acc = (
                    float(log_overall.iloc[0]["Accuracy (Overall)"] * 100.0)
                    if not log_overall.empty
                    else float("nan")
                )
                log_scalar("Train/Loss", log_loss, global_iter)
                log_scalar("Train/PixelAccuracy", log_acc, global_iter)
                log_scalar("Train/BaseLearnRate", current_lr, global_iter)
                _log_legacy_metric_scalars(
                    scalar_logger=log_scalar,
                    phase="Train",
                    loss_value=log_loss,
                    overall_df=log_overall,
                    class_df=log_class,
                    step=global_iter,
                )
                if log_last_preview is not None:
                    _log_legacy_preview_images(
                        writer=writer,
                        phase="Train",
                        image=log_last_preview["image"],
                        mask=log_last_preview["mask"],
                        pred_onehot=log_last_preview["pred_onehot"],
                        step=global_iter,
                    )
                log_conf.fill(0)
                log_loss_sum = 0.0
                log_batches = 0
                log_last_preview = None

            if global_iter % validation_interval == 0:
                v_loss, v_acc, v_overall, v_class, v_conf, v_preview = evaluate_loader(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=args.num_classes,
                    return_preview=True,
                )
                v_conf_norm = v_conf.div(v_conf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                if len(val_loss_hist) > 0:
                    val_loss_hist[-1] = float(v_loss)
                    val_acc_hist[-1] = float(v_acc)
                else:
                    val_loss_hist.append(float(v_loss))
                    val_acc_hist.append(float(v_acc))

                log_scalar("Validation/Loss", float(v_loss), global_iter)
                log_scalar("Validation/PixelAccuracy", float(v_acc), global_iter)
                _log_legacy_metric_scalars(
                    scalar_logger=log_scalar,
                    phase="Validation",
                    loss_value=float(v_loss),
                    overall_df=v_overall,
                    class_df=v_class,
                    step=global_iter,
                )
                _log_confusion_matrix_scalars(
                    scalar_logger=log_scalar,
                    phase="Validation",
                    conf_df=v_conf,
                    conf_norm_df=v_conf_norm,
                    step=global_iter,
                )
                if v_preview is not None:
                    _log_legacy_preview_images(
                        writer=writer,
                        phase="Validation",
                        image=v_preview["image"],
                        mask=v_preview["mask"],
                        pred_onehot=v_preview["pred_onehot"],
                        step=global_iter,
                    )
                run_monitor_evaluations(phase_prefix="MonitorValidation", step=global_iter)

                current_val_dice = float(
                    v_overall.iloc[0]["Dice (Overall)"] if not v_overall.empty else np.nan
                )
                log_scalar("Validation/DiceOverall", current_val_dice, global_iter)
                if np.isfinite(current_val_dice):
                    best_val_dice = max(best_val_dice, current_val_dice)

                if args.early_stopping_metric == "val_dice":
                    current_early_stop_score = current_val_dice
                    improved = np.isfinite(current_early_stop_score) and (
                        current_early_stop_score - best_early_stop_score
                    ) > float(args.early_stopping_min_delta)
                else:
                    current_early_stop_score = float(v_loss)
                    improved = (best_early_stop_score - current_early_stop_score) > float(
                        args.early_stopping_min_delta
                    )

                if np.isfinite(float(v_loss)):
                    best_val_loss = min(best_val_loss, float(v_loss))
                if improved:
                    best_early_stop_score = current_early_stop_score
                    bad_val_checks = 0
                    if bool(args.save_best_model):
                        torch.save(model.state_dict(), best_model_path)
                else:
                    bad_val_checks += 1

                if bad_val_checks >= int(args.validation_patience):
                    print(
                        f"Early stopping at epoch={epoch + 1}, iter={global_iter} "
                        f"(no {args.early_stopping_metric} improvement in {bad_val_checks} checks)."
                    )
                    stop_training = True
                    break

        if (epoch + 1) % save_model_interval == 0:
            torch.save(model.state_dict(), model_dir / f"model_epoch_{epoch + 1}.pth")

        if scheduler is not None:
            scheduler.step()

        if stop_training:
            break

    final_model_path = model_dir / "model_final.pth"
    torch.save(model.state_dict(), final_model_path)

    if bool(args.save_best_model) and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, overall_df, class_df, conf_df, test_preview = evaluate_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=args.num_classes,
        return_preview=True,
    )
    conf_norm_df = conf_df.div(conf_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
    overall_df.to_csv(analysis_dir / f"DataSetMetrics-{dt}.txt", sep="\t", index=False)
    class_df.to_csv(analysis_dir / f"ClassMetrics-{dt}.txt", sep="\t", index=False)
    conf_df.to_csv(analysis_dir / f"ConfusionMatrix-{dt}.txt", sep="\t")
    conf_norm_df.to_csv(analysis_dir / f"NormalizedConfusionMatrix-{dt}.txt", sep="\t")

    # MATLAB-like TrainingInfo struct export.
    info = {
        "TrainingLoss": np.asarray(train_loss_hist, dtype=np.float32),
        "ValidationLoss": np.asarray(val_loss_hist, dtype=np.float32),
        "TrainingAccuracy": np.asarray(train_acc_hist, dtype=np.float32),
        "ValidationAccuracy": np.asarray(val_acc_hist, dtype=np.float32),
        "BaseLearnRate": np.asarray(lr_hist, dtype=np.float32),
        "FinalValidationAccuracy": _last_finite(np.asarray(val_acc_hist), default=np.nan),
        "FinalValidationLoss": _last_finite(np.asarray(val_loss_hist), default=np.nan),
        "OutputNetworkIteration": int(global_iter),
        "FinalTestLoss": float(test_loss),
        "FinalTestPixelAccuracy": float(test_acc),
        "BestValidationLoss": float(best_val_loss) if np.isfinite(best_val_loss) else np.nan,
        "BestValidationDice": float(best_val_dice) if np.isfinite(best_val_dice) else np.nan,
        "BestEarlyStoppingScore": (
            float(best_early_stop_score) if np.isfinite(best_early_stop_score) else np.nan
        ),
        "EarlyStoppingMetric": str(args.early_stopping_metric),
    }

    if savemat is not None:
        savemat(experiment_path / f"TrainingInfo_{dt}.mat", {"info": info})

    summary = pd.DataFrame(
        [
            {
                "FinalValidationLoss": info["FinalValidationLoss"],
                "FinalValidationAccuracy": info["FinalValidationAccuracy"],
                "FinalTestLoss": float(test_loss),
                "FinalTestPixelAccuracy": float(test_acc),
                "OutputNetworkIteration": int(global_iter),
                "BestValidationLoss": float(best_val_loss) if np.isfinite(best_val_loss) else np.nan,
                "BestValidationDice": float(best_val_dice) if np.isfinite(best_val_dice) else np.nan,
                "BestEarlyStoppingScore": (
                    float(best_early_stop_score) if np.isfinite(best_early_stop_score) else np.nan
                ),
                "EarlyStoppingMetric": str(args.early_stopping_metric),
            }
        ]
    )
    summary.to_csv(analysis_dir / f"TrainingSummary-{dt}.csv", index=False)

    # Save a consolidated Excel workbook with final metrics and full training history.
    history_df = pd.DataFrame(
        {
            "Iteration": np.arange(1, len(train_loss_hist) + 1, dtype=np.int64),
            "TrainingLoss": np.asarray(train_loss_hist, dtype=np.float32),
            "ValidationLoss": np.asarray(val_loss_hist, dtype=np.float32),
            "TrainingAccuracy": np.asarray(train_acc_hist, dtype=np.float32),
            "ValidationAccuracy": np.asarray(val_acc_hist, dtype=np.float32),
            "BaseLearnRate": np.asarray(lr_hist, dtype=np.float32),
        }
    )
    history_df["ValidationChecked"] = history_df["ValidationLoss"].notna().astype(np.int32)

    log_scalar("Test/Loss", float(test_loss), global_iter)
    log_scalar("Test/PixelAccuracy", float(test_acc), global_iter)
    _log_legacy_metric_scalars(
        scalar_logger=log_scalar,
        phase="Test",
        loss_value=float(test_loss),
        overall_df=overall_df,
        class_df=class_df,
        step=global_iter,
    )
    _log_confusion_matrix_scalars(
        scalar_logger=log_scalar,
        phase="Test",
        conf_df=conf_df,
        conf_norm_df=conf_norm_df,
        step=global_iter,
    )
    if test_preview is not None:
        _log_legacy_preview_images(
            writer=writer,
            phase="Test",
            image=test_preview["image"],
            mask=test_preview["mask"],
            pred_onehot=test_preview["pred_onehot"],
            step=global_iter,
        )
    run_monitor_evaluations(phase_prefix="MonitorTest", step=global_iter)

    args_df = pd.DataFrame(
        [{"Argument": k, "Value": str(v)} for k, v in sorted(vars(args).items())]
    )
    scalar_df = pd.DataFrame(scalar_records, columns=["Step", "Tag", "Value"])
    monitor_df = pd.DataFrame(monitor_records)

    excel_path = analysis_dir / f"TrainingAllInOne-{dt}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer_xlsx:
            args_df.to_excel(writer_xlsx, sheet_name="RunConfig", index=False)
            summary.to_excel(writer_xlsx, sheet_name="TrainingSummary", index=False)
            history_df.to_excel(writer_xlsx, sheet_name="TrainingHistory", index=False)
            scalar_df.to_excel(writer_xlsx, sheet_name="ScalarLog", index=False)
            if not monitor_df.empty:
                monitor_df.to_excel(writer_xlsx, sheet_name="MonitorSeries", index=False)
            overall_df.to_excel(writer_xlsx, sheet_name="DataSetMetrics", index=False)
            class_df.to_excel(writer_xlsx, sheet_name="ClassMetrics", index=False)
            conf_df.to_excel(writer_xlsx, sheet_name="ConfusionMatrix", index=True)
            conf_norm_df.to_excel(
                writer_xlsx, sheet_name="NormalizedConfusion", index=True
            )
        print(f"Saved Excel report: {excel_path}")
    except Exception as e:
        print(f"[WARN] Failed to save Excel report: {e}")

    if not monitor_df.empty:
        monitor_df.to_csv(analysis_dir / f"MonitoredSessionMetrics-{dt}.csv", index=False)

    writer.close()

    print(f"Saved final model: {final_model_path}")
    if best_model_path.exists():
        print(f"Saved best model: {best_model_path}")
    print(f"Saved analysis: {analysis_dir}")


if __name__ == "__main__":
    main()
