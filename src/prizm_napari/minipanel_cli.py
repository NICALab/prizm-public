#!/usr/bin/env python3
"""
CLI for PRIZM mini-panel/heatmap/LDA analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prizm_napari.minipanel_analysis import run_minipanel_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run PRIZM mini-panel + heatmap + LDA/PCA/t-SNE analysis"
    )
    parser.add_argument("--data-dir", required=True, help="Directory with .xlsx files")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--control-group", default="CTRL", help="Control group name")
    parser.add_argument(
        "--ordered-files",
        default=None,
        help="Comma-separated file order to use instead of alphabetical auto-order",
    )
    parser.add_argument(
        "--reference-group",
        default=None,
        help="Reference group name for stats/heatmap (defaults to detected control group)",
    )
    parser.add_argument("--n-cols", type=int, default=5, help="Bar panel columns")
    parser.add_argument("--exclude-ctrl-heatmap", action="store_true", help="Do not include CTRL column in heatmap")
    parser.add_argument(
        "--save-all-pairs-excel",
        action="store_true",
        help="Save all pairwise Welch t-test results to the stats workbook",
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap generation")
    parser.add_argument("--no-lda", action="store_true", help="Skip LDA")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE")
    args = parser.parse_args()

    if not Path(args.data_dir).is_dir():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    ordered_files = None
    if args.ordered_files:
        ordered_files = [x.strip() for x in args.ordered_files.split(",") if x.strip()]

    try:
        result = run_minipanel_analysis(
            data_folder=args.data_dir,
            output_dir=args.output_dir,
            ordered_files=ordered_files,
            include_ctrl_in_heatmap=not args.exclude_ctrl_heatmap,
            make_heatmap=not args.no_heatmap,
            do_lda=not args.no_lda,
            do_pca=not args.no_pca,
            do_tsne=not args.no_tsne,
            n_cols=args.n_cols,
            control_group_name=args.control_group,
            reference_group_name=args.reference_group,
            save_all_pairs_excel=args.save_all_pairs_excel,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("PRIZM MiniPanel Analysis Completed")
    print("=" * 60)
    print(f"Output directory: {result['output_dir']}")
    print(f"Panel directory: {result['panel_dir']}")
    print(f"Stats file: {result['stats_xlsx']}")
    print(f"Groups: {result['n_groups']} | Params: {result['n_params']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
