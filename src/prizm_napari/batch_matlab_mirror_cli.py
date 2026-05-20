#!/usr/bin/env python3
"""
One-shot CLI:
1) Run PRIZM batch segmentation + feature extraction
2) Rebuild outputs into MATLAB-like condition layout
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from prizm_napari.batch_segmentation_cli import run_batch_segmentation
from prizm_napari.matlab_mirror import mirror_batch_output_to_matlab_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRIZM batch run with MATLAB-like mirrored output layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-dir", required=True, help="Root rawdata directory")
    parser.add_argument("--output-root", required=True, help="Final output root (MATLAB-like layout)")
    parser.add_argument("--model", required=True, help="Path to .pth model")

    parser.add_argument("--raw-batch-output", default=None, help="Intermediate core output directory")
    parser.add_argument(
        "--remove-raw-batch-output",
        action="store_true",
        help="Remove intermediate raw batch output after mirror rebuild",
    )
    parser.add_argument(
        "--no-merged-artifacts",
        action="store_true",
        help="Do not make merged jpg/gif artifacts in <condition>/merged",
    )

    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--encoder-depth", type=int, default=3)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--encoder-output-stride", type=int, default=8)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--atrous-rates", type=str, default="3 6 9")

    parser.add_argument("--metadata-mode", type=str, choices=["xml", "manual"], default="xml")
    parser.add_argument("--metadata-file", type=str, default=None)
    parser.add_argument("--resize-scale", type=float, default=None)
    parser.add_argument("--frame-interval", type=float, default=None)

    parser.set_defaults(infer_postprocess=False)
    parser.add_argument(
        "--infer-postprocess",
        action="store_true",
        dest="infer_postprocess",
        help="Enable infer-level postprocess filtering (off by default for MATLAB parity)",
    )
    parser.add_argument("--infer-batch-size", type=int, default=1)

    parser.set_defaults(use_amp=True)
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    model_path = Path(args.model)

    if not data_dir.is_dir():
        print(f"Error: data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"Error: model file does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    raw_batch_out = Path(args.raw_batch_output) if args.raw_batch_output else (output_root / "_raw_batch_cli")
    raw_batch_out.mkdir(parents=True, exist_ok=True)

    try:
        atrous_rates = [int(x) for x in args.atrous_rates.split()] if args.atrous_rates else []
    except ValueError:
        print(f"Error: invalid --atrous-rates value: {args.atrous_rates}", file=sys.stderr)
        sys.exit(1)

    use_metadata_xml = args.metadata_mode == "xml"
    if not use_metadata_xml and (args.resize_scale is None or args.frame_interval is None):
        print("Error: --resize-scale and --frame-interval are required when --metadata-mode manual", file=sys.stderr)
        sys.exit(1)

    print("[1/2] Running batch segmentation + feature extraction...")
    run_batch_segmentation(
        root_dir=str(data_dir),
        output_dir=str(raw_batch_out),
        model_path=str(model_path),
        channel=args.channel,
        grayscale=args.grayscale,
        backbone=args.backbone,
        encoder_depth=args.encoder_depth,
        decoder_channels=args.decoder_channels,
        encoder_output_stride=args.encoder_output_stride,
        input_channels=args.input_channels,
        atrous_rates=atrous_rates,
        infer_postprocess=args.infer_postprocess,
        resize_scale=args.resize_scale,
        frame_interval=args.frame_interval,
        metadata_file=args.metadata_file,
        use_metadata_xml=use_metadata_xml,
        infer_batch_size=args.infer_batch_size,
        use_amp=args.use_amp,
    )

    print("[2/2] Rebuilding MATLAB-like mirrored layout...")
    status_csv = mirror_batch_output_to_matlab_layout(
        data_root=data_dir,
        output_root=output_root,
        raw_batch_out=raw_batch_out,
        make_merged_artifacts=(not args.no_merged_artifacts),
    )

    if args.remove_raw_batch_output and raw_batch_out.exists():
        shutil.rmtree(raw_batch_out)
        print(f"[INFO] Removed intermediate raw batch output: {raw_batch_out}")

    print(f"[DONE] Output root: {output_root}")
    print(f"[DONE] Status CSV: {status_csv}")


if __name__ == "__main__":
    main()
