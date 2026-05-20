#!/usr/bin/env python3
"""
CLI for PRIZM 2-stage MoA prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prizm_napari.moa_analysis import run_full_moa_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run PRIZM 2-stage hierarchical MoA analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-dir", required=True, help="Training Excel directory")
    parser.add_argument("--unknown-dir", required=True, help="Unknown Excel directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--kfold", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--clip-z", type=float, default=6.0, help="Z-score clipping range")
    parser.add_argument(
        "--missing-frac-max",
        type=float,
        default=0.3,
        help="Maximum missing fraction per feature",
    )
    parser.add_argument(
        "--min-match-frac",
        type=float,
        default=0.90,
        help="Minimum feature-column match fraction against the vehicle reference",
    )
    parser.add_argument("--top-features", type=int, default=0, help="ANOVA top-N features (0=off)")
    parser.add_argument("--rng-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-trees", type=int, default=200, help="Number of trees for BAG model")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Stage1 target FPR")
    parser.add_argument("--stage1-final-id", default="S1_LOGI", help="Production Stage1 model ID")
    parser.add_argument(
        "--sim-metric",
        choices=["euclid", "cosine"],
        default="euclid",
        help="Similarity metric for similarity tables",
    )
    parser.add_argument("--sim-top-k", type=int, default=3, help="Top-K similarity groups to report")
    parser.add_argument("--self-label", default="Self", help="Label used for the self-similarity reference")
    parser.add_argument("--dominance-alpha", type=float, default=0.05, help="Alpha used for dominance statistics")
    parser.add_argument(
        "--dominance-competitor-mode",
        choices=["mean", "top2mean", "best"],
        default="mean",
        help="Competitor baseline used in dominance statistics",
    )
    parser.add_argument("--perm-n", type=int, default=10000, help="Monte Carlo sign-flip permutations for dominance")
    parser.add_argument(
        "--perm-max-exact-n",
        type=int,
        default=16,
        help="Use exact sign-flip enumeration when paired sample count is at most this value",
    )
    parser.add_argument("--perm-seed", type=int, default=0, help="Permutation RNG seed for dominance statistics")
    parser.add_argument("--tost-alpha", type=float, default=0.05, help="Legacy no-op, retained for compatibility")
    parser.add_argument(
        "--tost-delta-softmax",
        type=float,
        default=0.15,
        help="Legacy no-op, retained for compatibility",
    )
    parser.add_argument(
        "--tost-delta-distance",
        type=float,
        default=1.18,
        help="Legacy no-op, retained for compatibility",
    )
    parser.add_argument(
        "--sim-multiplier",
        type=float,
        default=1.5,
        help="Legacy no-op, retained for compatibility",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip visual report generation",
    )
    parser.add_argument(
        "--no-robust-control-stats",
        action="store_true",
        help="Use mean/std instead of median/MAD for control normalization",
    )
    parser.add_argument(
        "--no-self-similarity",
        action="store_true",
        help="Disable adding the current file as a similarity reference",
    )
    parser.add_argument(
        "--no-dominance-stats",
        action="store_true",
        help="Disable saving similarity-based dominance statistics",
    )
    parser.add_argument(
        "--no-ml-dominance-stats",
        action="store_true",
        help="Disable saving ML-similarity dominance statistics",
    )
    parser.add_argument(
        "--include-self-in-dominance",
        action="store_true",
        help="Include Self as a dominance competitor/target instead of excluding it",
    )
    parser.add_argument(
        "--no-tost-vs-self",
        action="store_true",
        help="Legacy no-op, retained for compatibility",
    )
    parser.add_argument(
        "--no-train-in-analysis",
        action="store_true",
        help="Do not include training files in the prediction outputs",
    )
    args = parser.parse_args()

    if not Path(args.train_dir).is_dir():
        print(f"Error: train directory not found: {args.train_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.unknown_dir).is_dir():
        print(f"Error: unknown directory not found: {args.unknown_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_full_moa_analysis(
            train_dir=args.train_dir,
            unknown_dir=args.unknown_dir,
            out_dir=args.output_dir,
            use_robust_control_stats=not args.no_robust_control_stats,
            clip_z_value=args.clip_z,
            missing_frac_max=args.missing_frac_max,
            n_top_features=args.top_features,
            kfold_fish=args.kfold,
            rng_seed=args.rng_seed,
            n_trees=args.n_trees,
            target_fpr=args.target_fpr,
            make_figures=not args.no_figures,
            stage1_final_id=args.stage1_final_id,
            sim_metric=args.sim_metric,
            sim_top_k=args.sim_top_k,
            include_self_in_similarity=not args.no_self_similarity,
            self_similarity_label=args.self_label,
            save_dominance_stats=not args.no_dominance_stats,
            dominance_alpha=args.dominance_alpha,
            exclude_self_in_dominance=not args.include_self_in_dominance,
            save_dominance_stats_ml=not args.no_ml_dominance_stats,
            dominance_competitor_mode=args.dominance_competitor_mode,
            perm_n=args.perm_n,
            perm_max_exact_n=args.perm_max_exact_n,
            perm_seed=args.perm_seed,
            save_tost_vs_self=not args.no_tost_vs_self,
            tost_alpha=args.tost_alpha,
            tost_delta_softmax=args.tost_delta_softmax,
            tost_delta_distance=args.tost_delta_distance,
            sim_multiplier=args.sim_multiplier,
            include_train_in_analysis=not args.no_train_in_analysis,
            min_match_frac=args.min_match_frac,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("PRIZM 2-Stage MoA Completed")
    print("=" * 60)
    print(f"Output directory: {result['out_dir']}")
    print(f"Bundle: {result['bundle_path']}")
    print(f"Train report: {result['train_report_xlsx']}")
    print(f"Master: {result['master_xlsx']}")
    print(f"Unknown files processed: {result['n_unknown_files']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
