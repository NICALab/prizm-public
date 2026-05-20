#!/usr/bin/env python3
"""
Command-line interface for PRIZM chemical analysis.
"""

import argparse
import sys
from pathlib import Path
import importlib.util

# Import directly from module to avoid loading widget dependencies
# Use importlib to avoid triggering __init__.py imports
spec = importlib.util.spec_from_file_location(
    "chemical_analysis",
    Path(__file__).parent / "chemical_analysis.py"
)
chemical_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chemical_analysis)
run_full_analysis = chemical_analysis.run_full_analysis


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Analyze chemical analysis data from PRIZM video analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prizm-chemical-analysis \\
    --data-dir /path/to/data \\
    --output-dir /path/to/output
  
  prizm-chemical-analysis \\
    --data-dir /path/to/data \\
    --output-dir /path/to/output \\
    --n-clusters 5 \\
    --alpha 0.01
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing CSV files with functional feature vectors'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results (figures, tables, report)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters for K-means (default: auto-determined)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
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
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    try:
        results = run_full_analysis(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            n_clusters=args.n_clusters,
            alpha=args.alpha,
            random_state=args.random_state
        )
        
        print("\n" + "="*60)
        print("Analysis Summary")
        print("="*60)
        print(f"Total samples: {len(results['data'])}")
        print(f"Number of clusters: {results['cluster_results']['n_clusters']}")
        if not results['anova_results'].empty and 'significant' in results['anova_results'].columns:
            n_sig = results['anova_results']['significant'].sum()
            print(f"Significant features (ANOVA): {n_sig}")
        else:
            print("Significant features (ANOVA): N/A")
        print(f"\nResults saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
