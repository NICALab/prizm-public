#!/usr/bin/env python3
"""
Command-line interface for PRIZM batch segmentation.
"""

import argparse
import sys
from pathlib import Path
import importlib.util
from tqdm import tqdm

# Import the shared core function (no GUI dependencies)
from prizm_napari.batch_segmentation_core import run_batch_segmentation_core


def run_batch_segmentation(
    root_dir: str,
    output_dir: str,
    model_path: str,
    model_type: str = "onnx",
    channel: int = 1,
    grayscale: bool = False,
    backbone: str = "resnet50",
    encoder_depth: int = 3,
    decoder_channels: int = 256,
    encoder_output_stride: int = 8,
    atrous_rates: list = None,
    input_channels: int = 1,
    infer_postprocess: bool = False,
    resize_scale: float = None,
    frame_interval: float = None,
    metadata_file: str = None,
    use_metadata_xml: bool = True,
    infer_batch_size: int = 1,
    use_amp: bool = True,
):
    """
    Run batch segmentation and analysis on organized directory structure.
    
    This function calls the exact same core function that the GUI uses.
    
    Expected directory structure:
        {CHEMICAL_TYPE}_{CONCENTRATION}/
            sample_{SAMPLE_ID}/
                frame_{FRAME_ID}.png (or .tif, .jpg, etc.)
                metadata/ (optional)
                    {SAMPLE_ID}_Properties.xml
    """
    import os
    
    # Convert Path objects to strings for compatibility
    root_dir = str(root_dir)
    output_dir = str(output_dir)
    
    if atrous_rates is None:
        atrous_rates = []
    
    # Call the exact same core function that the GUI uses
    # The core function now handles all progress tracking with tqdm internally
    batch_combined_df, _, _ = run_batch_segmentation_core(
        root_dir=root_dir,
        out_dir=output_dir,
        model_path=model_path,
        model_type=model_type,
        channel=channel,
        grayscale=grayscale,
        backbone=backbone,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        encoder_output_stride=encoder_output_stride,
        atrous_rates=atrous_rates,
        input_channels=input_channels,
        meta_manual=not use_metadata_xml,
        resize_scale=resize_scale,
        frame_interval=frame_interval,
        metadata_file=metadata_file,
        load_to_viewer=False,
        save_analysis_vis=False,
        infer_postprocess=infer_postprocess,
        infer_batch_size=infer_batch_size,
        use_amp=use_amp,
        progress_callback=None,  # tqdm handles progress now
    )
    
    print(f"\nBatch processing complete!")
    print(f"Processed {len(batch_combined_df)} rows in combined results")
    
    return batch_combined_df


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Batch segmentation and analysis for PRIZM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prizm-batch-segmentation \\
    --data-dir /path/to/data \\
    --output-dir /path/to/output \\
    --model /path/to/model.onnx \\
    --model-type onnx
  
  prizm-batch-segmentation \\
    --data-dir /path/to/data \\
    --output-dir /path/to/output \\
    --model /path/to/model.pth \\
    --model-type pth \\
    --channel 1 \\
    --backbone resnet50 \\
    --encoder-depth 5 \\
    --decoder-channels 256
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Root data directory with structure: {CHEMICAL_TYPE}_{CONCENTRATION}/sample_{SAMPLE_ID}/frame_{FRAME_ID}.png'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to segmentation model file (.pth PyTorch checkpoint or .onnx exported model)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['auto', 'onnx', 'pth'],
        default='onnx',
        help='Model backend to use (default: onnx)'
    )
    
    parser.add_argument(
        '--channel',
        type=int,
        default=1,
        help='Channel to segment (default: 1 = green)'
    )
    
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Convert images to grayscale'
    )

    parser.set_defaults(infer_postprocess=False)
    parser.add_argument(
        '--postprocess-masks',
        action='store_true',
        dest='infer_postprocess',
        help='Enable infer-time mask postprocessing before saving masks and running downstream analysis'
    )
    parser.add_argument(
        '--infer-postprocess',
        action='store_true',
        dest='infer_postprocess',
        help='Compatibility alias for --postprocess-masks'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50',
        help='Model backbone (default: resnet50)'
    )
    
    parser.add_argument(
        '--encoder-depth',
        type=int,
        default=3,
        help='Encoder depth (default: 3)'
    )
    
    parser.add_argument(
        '--decoder-channels',
        type=int,
        default=256,
        help='Decoder channels (default: 256)'
    )
    
    parser.add_argument(
        '--encoder-output-stride',
        type=int,
        default=8,
        help='Encoder output stride (default: 8)'
    )

    parser.add_argument(
        '--input-channels',
        type=int,
        default=1,
        help='Model input channels (must match checkpoint training config).'
    )
    
    parser.add_argument(
        '--atrous-rates',
        type=str,
        default='',
        help='Atrous rates as space-separated integers (e.g., "3 6 9")'
    )
    
    parser.add_argument(
        '--metadata-mode',
        type=str,
        choices=['xml', 'manual'],
        default='xml',
        help='Metadata mode: xml (use XML files) or manual (use provided values)'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        default=None,
        help='Path to metadata XML file (if using single file for all samples)'
    )
    
    parser.add_argument(
        '--resize-scale',
        type=float,
        default=None,
        help='Resize scale for manual metadata mode (e.g., 0.5)'
    )
    
    parser.add_argument(
        '--frame-interval',
        type=float,
        default=None,
        help='Frame interval in seconds for manual metadata mode (e.g., 0.062)'
    )

    parser.add_argument(
        '--infer-batch-size',
        type=int,
        default=1,
        help='Inference batch size (default: 1). Increase cautiously if you have spare GPU VRAM.'
    )

    parser.set_defaults(use_amp=True)
    parser.add_argument(
        '--no-amp',
        action='store_false',
        dest='use_amp',
        help='Disable mixed-precision inference (AMP).'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"Error: Data path is not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse atrous rates
    atrous_rates = []
    if args.atrous_rates:
        try:
            atrous_rates = [int(r) for r in args.atrous_rates.split()]
        except ValueError:
            print(f"Warning: Invalid atrous rates format: {args.atrous_rates}", file=sys.stderr)
            atrous_rates = []
    
    # Validate manual metadata mode
    use_metadata_xml = (args.metadata_mode == 'xml')
    if not use_metadata_xml:
        if args.resize_scale is None or args.frame_interval is None:
            print("Error: --resize-scale and --frame-interval are required in manual metadata mode", file=sys.stderr)
            sys.exit(1)
    
    # Run batch segmentation
    try:
        run_batch_segmentation(
            root_dir=str(data_dir),
            output_dir=str(output_dir),
            model_path=str(model_path),
            model_type=args.model_type,
            channel=args.channel,
            grayscale=args.grayscale,
            backbone=args.backbone,
            encoder_depth=args.encoder_depth,
            decoder_channels=args.decoder_channels,
            encoder_output_stride=args.encoder_output_stride,
            atrous_rates=atrous_rates,
            input_channels=args.input_channels,
            resize_scale=args.resize_scale,
            frame_interval=args.frame_interval,
            metadata_file=args.metadata_file,
            use_metadata_xml=use_metadata_xml,
            infer_postprocess=args.infer_postprocess,
            infer_batch_size=args.infer_batch_size,
            use_amp=args.use_amp,
        )
    except Exception as e:
        print(f"Error during batch segmentation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
