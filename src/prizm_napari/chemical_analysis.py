"""
Core analysis functions for chemical analysis data.
No napari dependencies - can be used in CLI or napari widget.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Analysis imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import umap

# Statistical imports
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from prizm_napari.plot_colors import distinct_categorical_colors

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def _distinct_color_lookup(labels) -> Dict:
    unique_labels = list(dict.fromkeys(labels))
    palette = distinct_categorical_colors(len(unique_labels))
    return {label: palette[i] for i, label in enumerate(unique_labels)}


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse standardized filename to extract metadata.
    Pattern: {DATE}_{CHEMICAL_TYPE}_{CONCENTRATION}_{SESSION_ID}.csv
    """
    name = filename.replace('.csv', '').replace('.xlsx', '')
    parts = name.split('_')
    
    if len(parts) >= 4:
        date = parts[0]
        chemical_type = parts[1]
        concentration = parts[2]
        session_id = '_'.join(parts[3:])  # Handle session IDs with underscores
        return {
            'date': date,
            'chemical_type': chemical_type,
            'concentration': concentration,
            'session_id': session_id
        }
    return {}


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from directory and combine into single DataFrame.
    Extracts metadata from 'File Name' column in each CSV file.
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Parse metadata from 'File Name' column if it exists
            if 'File Name' in df.columns:
                # Extract metadata from each row's File Name
                metadata_list = []
                for filename in df['File Name']:
                    metadata = parse_filename(str(filename))
                    metadata_list.append(metadata)
                
                # Add metadata columns
                metadata_df = pd.DataFrame(metadata_list)
                for col in metadata_df.columns:
                    df[col] = metadata_df[col]
            else:
                # Fallback: try to parse from filename
                metadata = parse_filename(csv_file.name)
                for key, value in metadata.items():
                    df[key] = value
            
            # Add source file
            df['source_file'] = csv_file.name
            
            all_data.append(df)
        except Exception as e:
            warnings.warn(f"Error loading {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("No data files could be loaded")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure metadata columns exist
    for col in ['date', 'chemical_type', 'concentration', 'session_id']:
        if col not in combined_df.columns:
            combined_df[col] = 'Unknown'
    
    return combined_df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare feature matrix for analysis.
    Excludes metadata columns and 'File Name'.
    """
    # Get feature columns (exclude metadata and File Name)
    exclude_cols = ['File Name', 'date', 'chemical_type', 'concentration', 
                    'session_id', 'source_file']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Handle missing values - fill with median
    X = X.fillna(X.median())
    
    # Convert to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    # Convert to numpy array
    X_array = X.values
    
    return X, X_array, feature_cols


def perform_clustering(X: np.ndarray, n_clusters: Optional[int] = None, 
                      random_state: int = 42) -> Dict:
    """
    Perform K-means clustering and determine optimal number of clusters.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        silhouette_scores = []
        K_range = range(2, min(11, len(X) // 2))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append((k, score))
        
        if silhouette_scores:
            n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        else:
            n_clusters = 2
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'scaler': scaler,
        'model': kmeans,
        'X_scaled': X_scaled
    }


def perform_dimensionality_reduction(X: np.ndarray, method: str = 'umap',
                                    n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """
    Perform dimensionality reduction (PCA, t-SNE, or UMAP).
    """
    n_samples = len(X)
    # Adjust n_components if we have fewer samples than requested components
    actual_n_components = min(n_components, n_samples, X.shape[1])
    
    if actual_n_components < n_components:
        warnings.warn(f"Reducing n_components from {n_components} to {actual_n_components} "
                     f"(n_samples={n_samples}, n_features={X.shape[1]})")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'pca':
        reducer = PCA(n_components=actual_n_components, random_state=random_state)
        X_reduced = reducer.fit_transform(X_scaled)
        # Pad with zeros if we needed fewer components
        if actual_n_components < n_components:
            padding = np.zeros((X_reduced.shape[0], n_components - actual_n_components))
            X_reduced = np.hstack([X_reduced, padding])
    elif method == 'tsne':
        if n_samples < 2:
            # t-SNE needs at least 2 samples
            X_reduced = np.zeros((n_samples, n_components))
        else:
            reducer = TSNE(n_components=actual_n_components, random_state=random_state, 
                      perplexity=min(30, max(1, len(X) - 1)))
            X_reduced = reducer.fit_transform(X_scaled)
            # Pad with zeros if needed
            if actual_n_components < n_components:
                padding = np.zeros((X_reduced.shape[0], n_components - actual_n_components))
                X_reduced = np.hstack([X_reduced, padding])
    elif method == 'umap':
        if n_samples < 2:
            # UMAP needs at least 2 samples
            X_reduced = np.zeros((n_samples, n_components))
        else:
            reducer = umap.UMAP(n_components=actual_n_components, random_state=random_state)
            X_reduced = reducer.fit_transform(X_scaled)
            # Pad with zeros if needed
            if actual_n_components < n_components:
                padding = np.zeros((X_reduced.shape[0], n_components - actual_n_components))
                X_reduced = np.hstack([X_reduced, padding])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_reduced


def perform_statistical_tests(df: pd.DataFrame, feature_cols: List[str], 
                              group_col: str = 'chemical_type', 
                              alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform ANOVA and post-hoc tests for each feature across groups.
    """
    results = []
    
    for feature in feature_cols:
        # Get groups
        groups = [group[feature].dropna().values 
                 for name, group in df.groupby(group_col)]
        group_names = [name for name, _ in df.groupby(group_col)]
        
        if len(groups) < 2:
            continue
        
        # Remove groups with < 2 samples
        valid_groups = [(g, name) for g, name in zip(groups, group_names) if len(g) >= 2]
        if len(valid_groups) < 2:
            continue
        
        groups, group_names = zip(*valid_groups)
        
        # ANOVA
        try:
            f_stat, p_value = f_oneway(*groups)
        except:
            f_stat, p_value = np.nan, np.nan
        
        # Post-hoc tests (Tukey HSD)
        tukey_results = None
        if not np.isnan(p_value) and p_value < alpha and len(groups) > 2:
            try:
                # Prepare data for Tukey
                data_list = []
                group_list = []
                for g, name in zip(groups, group_names):
                    data_list.extend(g)
                    group_list.extend([name] * len(g))
                
                tukey_df = pd.DataFrame({'value': data_list, 'group': group_list})
                tukey_results = pairwise_tukeyhsd(tukey_df['value'], tukey_df['group'], alpha=alpha)
            except:
                pass
        
        results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha if not np.isnan(p_value) else False,
            'n_groups': len(groups),
            'group_names': ', '.join(group_names)
        })
    
    return pd.DataFrame(results)


def perform_pairwise_tests(df: pd.DataFrame, feature_cols: List[str],
                          group_col: str = 'chemical_type',
                          alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform pairwise t-tests between all groups for each feature.
    """
    results = []
    groups = df[group_col].unique()
    
    for feature in feature_cols:
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                data1 = df[df[group_col] == group1][feature].dropna()
                data2 = df[df[group_col] == group2][feature].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Welch's t-test (unequal variances)
                try:
                    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    results.append({
                        'feature': feature,
                        'group1': group1,
                        'group2': group2,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < alpha,
                        'mean_group1': data1.mean(),
                        'mean_group2': data2.mean(),
                        'n_group1': len(data1),
                        'n_group2': len(data2)
                    })
                except:
                    pass
    
    return pd.DataFrame(results)


def calculate_summary_statistics(df: pd.DataFrame, feature_cols: List[str],
                                group_col: str = 'chemical_type') -> pd.DataFrame:
    """
    Calculate summary statistics for each feature by group.
    """
    summary = []
    
    for feature in feature_cols:
        for group_name, group_df in df.groupby(group_col):
            values = group_df[feature].dropna()
            
            if len(values) == 0:
                continue
            
            summary.append({
                'feature': feature,
                'group': group_name,
                'n': len(values),
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'sem': values.sem() if len(values) > 1 else 0,
                'min': values.min(),
                'max': values.max(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            })
    
    return pd.DataFrame(summary)


def calculate_feature_importance(df: pd.DataFrame, feature_cols: List[str],
                                target_col: str = 'chemical_type',
                                random_state: int = 42) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.
    """
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0 or len(y.unique()) < 2:
        return pd.DataFrame()
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def create_visualizations(df: pd.DataFrame, X_reduced: np.ndarray, 
                         cluster_results: Dict, feature_cols: List[str],
                         output_dir: str, X_array: Optional[np.ndarray] = None) -> Tuple[Dict[str, str], Dict]:
    """
    Create all visualization figures and save to output directory.
    Returns tuple of (figure_paths, raw_data_dict) where raw_data_dict contains
    all computed data used in visualizations.
    """
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = {}
    raw_data = {
        'X_pca': None,
        'X_tsne': None,
        'mean_vectors': None,
        'mean_metadata': None,
        'mean_pca': None,
        'mean_umap': None,
        'mean_tsne': None,
        'corr_matrix': None
    }
    chemical_types = sorted(df['chemical_type'].unique(), key=lambda x: str(x))
    concentrations = sorted(df['concentration'].unique(), key=lambda x: str(x))
    chemical_type_color_map = _distinct_color_lookup(chemical_types)
    concentration_color_map = _distinct_color_lookup(concentrations)
    
    # 1. UMAP clustering colored by chemical type
    fig, ax = plt.subplots(figsize=(10, 8))
    for chemical_type in chemical_types:
        mask = df['chemical_type'] == chemical_type
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=chemical_type,
            alpha=0.8,
            s=50,
            c=[chemical_type_color_map[chemical_type]],
            edgecolors='black',
            linewidths=0.45,
        )
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP Visualization Colored by Chemical Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    path = figures_dir / 'clustering_umap_by_chemical_type.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['umap_chemical'] = str(path)
    
    # 2. UMAP clustering colored by concentration
    fig, ax = plt.subplots(figsize=(10, 8))
    for conc in concentrations:
        mask = df['concentration'] == conc
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=conc,
            alpha=0.8,
            s=50,
            c=[concentration_color_map[conc]],
            edgecolors='black',
            linewidths=0.45,
        )
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP Visualization Colored by Concentration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    path = figures_dir / 'clustering_umap_by_concentration.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['umap_concentration'] = str(path)
    
    # 3. K-means clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = cluster_results['labels']
    cluster_ids = sorted(np.unique(labels).tolist())
    cluster_color_map = _distinct_color_lookup(cluster_ids)
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=f'Cluster {cluster_id}',
            alpha=0.8,
            s=50,
            c=[cluster_color_map[cluster_id]],
            edgecolors='black',
            linewidths=0.45,
        )
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'K-means Clustering (k={cluster_results["n_clusters"]})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    path = figures_dir / 'clustering_kmeans.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['kmeans'] = str(path)
    
    # 4. Box plots for key features by chemical type
    key_features = feature_cols[:min(12, len(feature_cols))]  # Top 12 features
    n_cols = 4
    n_rows = (len(key_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        df.boxplot(column=feature, by='chemical_type', ax=ax)
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Remove extra subplots
    for idx in range(len(key_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Feature Distributions by Chemical Type', y=1.02)
    plt.tight_layout()
    path = figures_dir / 'boxplots_by_chemical_type.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['boxplots_chemical'] = str(path)
    
    # 5. Dose-response curves (for BaP with numeric concentrations)
    bap_df = df[df['chemical_type'] == 'BaP'].copy()
    if len(bap_df) > 0:
        # Try to convert concentrations to numeric
        bap_df['conc_numeric'] = pd.to_numeric(bap_df['concentration'], errors='coerce')
        bap_df = bap_df[bap_df['conc_numeric'].notna()]
        
        if len(bap_df) > 0 and len(bap_df['conc_numeric'].unique()) > 2:
            key_features = feature_cols[:min(6, len(feature_cols))]
            n_cols = 3
            n_rows = (len(key_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, feature in enumerate(key_features):
                ax = axes[idx]
                for conc in sorted(bap_df['conc_numeric'].unique()):
                    values = bap_df[bap_df['conc_numeric'] == conc][feature].dropna()
                    if len(values) > 0:
                        ax.scatter([conc] * len(values), values, alpha=0.5, s=30)
                        ax.errorbar(conc, values.mean(), yerr=values.sem(), 
                                   fmt='o', capsize=5, markersize=8)
                ax.set_xlabel('Concentration')
                ax.set_ylabel(feature)
                ax.set_title(feature)
                ax.grid(True, alpha=0.3)
            
            for idx in range(len(key_features), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.suptitle('Dose-Response Curves (BaP)', y=1.02)
            plt.tight_layout()
            path = figures_dir / 'dose_response_curves.png'
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            figure_paths['dose_response'] = str(path)
    
    # 6. Correlation heatmap
    corr_features = feature_cols[:min(20, len(feature_cols))]  # Limit for readability
    corr_matrix = df[corr_features].corr()
    raw_data['corr_matrix'] = corr_matrix
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    path = figures_dir / 'correlation_heatmap.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['correlation'] = str(path)
    
    # 7. PCA biplot
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    raw_data['X_pca'] = X_pca
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for chemical_type in chemical_types:
        mask = df['chemical_type'] == chemical_type
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=chemical_type,
            alpha=0.8,
            s=50,
            c=[chemical_type_color_map[chemical_type]],
            edgecolors='black',
            linewidths=0.45,
        )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Biplot Colored by Chemical Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    path = figures_dir / 'pca_biplot.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    figure_paths['pca'] = str(path)
    
    # 8. PCA, UMAP, t-SNE with distinct color per chemical_type/concentration combination
    if X_array is not None:
        combo_labels = []
        for chemical_type in chemical_types:
            concs_for_type = sorted(
                df[df['chemical_type'] == chemical_type]['concentration'].unique(),
                key=lambda x: str(x),
            )
            for conc in concs_for_type:
                combo_labels.append((chemical_type, conc))
        combo_color_map = _distinct_color_lookup(combo_labels)
        
        # Create PCA, UMAP, t-SNE plots
        methods = [
            ('PCA', X_pca, pca.explained_variance_ratio_),
            ('UMAP', X_reduced, None),
            ('t-SNE', None, None)
        ]
        
        # Compute t-SNE if needed
        X_tsne = None
        if X_array is not None:
            print("Computing t-SNE...")
            X_tsne = perform_dimensionality_reduction(X_array, method='tsne', 
                                                      n_components=2, random_state=42)
            raw_data['X_tsne'] = X_tsne
            methods[2] = ('t-SNE', X_tsne, None)
        
        for method_name, coords, variance_info in methods:
            if coords is None:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot each chemical type/concentration combination with a distinct color
            for chemical_type in chemical_types:
                mask = df['chemical_type'] == chemical_type
                
                for conc in sorted(df[mask]['concentration'].unique(), key=lambda x: str(x)):
                    conc_mask = mask & (df['concentration'] == conc)
                    if conc_mask.sum() > 0:
                        label = f'{chemical_type} ({conc})' if len(chemical_types) > 1 or len(df[mask]['concentration'].unique()) > 1 else chemical_type
                        ax.scatter(
                            coords[conc_mask, 0],
                            coords[conc_mask, 1],
                            c=[combo_color_map[(chemical_type, conc)]],
                            label=label,
                            alpha=0.8,
                            s=60,
                            edgecolors='black',
                            linewidths=0.5,
                        )
            
            # Set labels
            if method_name == 'PCA' and variance_info is not None:
                ax.set_xlabel(f'{method_name} 1 ({variance_info[0]:.1%} variance)', fontsize=12)
                ax.set_ylabel(f'{method_name} 2 ({variance_info[1]:.1%} variance)', fontsize=12)
            else:
                ax.set_xlabel(f'{method_name} 1', fontsize=12)
                ax.set_ylabel(f'{method_name} 2', fontsize=12)
            
            ax.set_title(
                f'{method_name} Visualization\nColor: Distinct chemical type + concentration combination',
                        fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            filename = f'{method_name.lower()}_chemical_type_saturation.png'
            path = figures_dir / filename
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            figure_paths[f'{method_name.lower()}_saturation'] = str(path)
    
    # 9. PCA, UMAP, t-SNE with mean feature vectors per chemical type + concentration combination
    if X_array is not None:
        # Calculate mean feature vector for each unique (chemical_type, concentration) combination
        mean_vectors = []
        mean_metadata = []
        
        # Group by chemical_type and concentration
        grouped = df.groupby(['chemical_type', 'concentration'])
        
        for (chemical_type, concentration), group_data in grouped:
            # Get mean feature vector for this group
            mean_vec = group_data[feature_cols].fillna(0).mean().values
            mean_vectors.append(mean_vec)
            
            # Get metadata
            mean_metadata.append({
                'chemical_type': chemical_type,
                'concentration': concentration,
                'n_samples': len(group_data),
                'label': f'{chemical_type}_{concentration}'
            })
        
        mean_vectors_array = np.array(mean_vectors)
        mean_df = pd.DataFrame(mean_metadata)
        raw_data['mean_vectors'] = mean_vectors_array
        raw_data['mean_metadata'] = mean_df
        
        # Only compute mean vector visualizations if we have at least 2 unique combinations
        if len(mean_vectors_array) >= 2:
            # Perform dimensionality reduction on mean vectors
            print("Computing PCA, UMAP, t-SNE for mean vectors...")
            mean_pca_coords = perform_dimensionality_reduction(mean_vectors_array, method='pca', 
                                                              n_components=2, random_state=42)
            mean_umap_coords = perform_dimensionality_reduction(mean_vectors_array, method='umap', 
                                                                n_components=2, random_state=42)
            mean_tsne_coords = perform_dimensionality_reduction(mean_vectors_array, method='tsne', 
                                                                 n_components=2, random_state=42)
            raw_data['mean_pca'] = mean_pca_coords
            raw_data['mean_umap'] = mean_umap_coords
            raw_data['mean_tsne'] = mean_tsne_coords
            
            # Get PCA explained variance
            scaler_pca = StandardScaler()
            mean_scaled = scaler_pca.fit_transform(mean_vectors_array)
            n_comp = min(2, len(mean_vectors_array), mean_vectors_array.shape[1])
            pca_mean = PCA(n_components=n_comp)
            pca_mean.fit(mean_scaled)
            pca_variance = pca_mean.explained_variance_ratio_
            # Pad variance if needed
            if len(pca_variance) < 2:
                pca_variance = np.pad(pca_variance, (0, 2 - len(pca_variance)), mode='constant', constant_values=0)
            
            combo_labels_mean = [
                (row['chemical_type'], row['concentration']) for _, row in mean_df.iterrows()
            ]
            combo_color_map_mean = _distinct_color_lookup(combo_labels_mean)
            
            # Create plots for each method
            methods_mean = [
                ('PCA', mean_pca_coords, pca_variance),
                ('UMAP', mean_umap_coords, None),
                ('t-SNE', mean_tsne_coords, None)
            ]
            
            for method_name, coords, variance_info in methods_mean:
                if coords is None:
                    continue
                
                fig, ax = plt.subplots(figsize=(12, 10))

                for idx, row in mean_df.iterrows():
                    chemical_type = row['chemical_type']
                    concentration = row['concentration']
                    label = f'{chemical_type} ({concentration})'
                    size = 200 + row['n_samples'] * 2
                    ax.scatter(
                        coords[idx, 0],
                        coords[idx, 1],
                        c=[combo_color_map_mean[(chemical_type, concentration)]],
                        label=label,
                        alpha=0.85,
                        s=size,
                        edgecolors='black',
                        linewidths=1.5,
                        zorder=3,
                    )
                    label_text = f"{row['chemical_type']}_{row['concentration']}"
                    ax.annotate(
                        label_text,
                        (coords[idx, 0], coords[idx, 1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8,
                        zorder=4,
                    )

                if method_name == 'PCA' and variance_info is not None:
                    ax.set_xlabel(f'{method_name} 1 ({variance_info[0]:.1%} variance)', fontsize=12)
                    ax.set_ylabel(f'{method_name} 2 ({variance_info[1]:.1%} variance)', fontsize=12)
                else:
                    ax.set_xlabel(f'{method_name} 1', fontsize=12)
                    ax.set_ylabel(f'{method_name} 2', fontsize=12)
                
                ax.set_title(f'{method_name} Visualization - Mean Feature Vectors\n'
                            f'Color: Distinct chemical type + concentration combination\n'
                            f'Marker Size: Number of Samples',
                            fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                filename = f'{method_name.lower()}_mean_vectors.png'
                path = figures_dir / filename
                plt.savefig(path, bbox_inches='tight', dpi=300)
                plt.close()
                figure_paths[f'{method_name.lower()}_mean'] = str(path)
        else:
            print(f"Skipping mean vector visualizations: only {len(mean_vectors_array)} unique chemical_type/concentration combination(s) found (need at least 2)")
            raw_data['mean_pca'] = None
            raw_data['mean_umap'] = None
            raw_data['mean_tsne'] = None
    
    return figure_paths, raw_data


def save_raw_data(df: pd.DataFrame, X_array: np.ndarray, X_umap: np.ndarray,
                 X_pca: Optional[np.ndarray], X_tsne: Optional[np.ndarray],
                 cluster_labels: np.ndarray, feature_cols: List[str],
                 mean_vectors: Optional[np.ndarray] = None,
                 mean_metadata: Optional[pd.DataFrame] = None,
                 mean_pca: Optional[np.ndarray] = None,
                 mean_umap: Optional[np.ndarray] = None,
                 mean_tsne: Optional[np.ndarray] = None,
                 corr_matrix: Optional[pd.DataFrame] = None,
                 output_dir: str = None) -> Dict[str, str]:
    """
    Save all raw data used in analysis figures.
    """
    data_dir = Path(output_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_paths = {}
    
    # Save combined data (already saved in save_tables, but include here for completeness)
    data_path = data_dir / 'combined_data.csv'
    df.to_csv(data_path, index=False)
    data_paths['combined_data'] = str(data_path)
    
    # Save UMAP coordinates
    umap_df = pd.DataFrame(X_umap, columns=['UMAP_1', 'UMAP_2'])
    umap_df = pd.concat([df[['File Name', 'chemical_type', 'concentration', 'session_id', 'source_file']].reset_index(drop=True),
                        umap_df], axis=1)
    umap_path = data_dir / 'umap_coordinates.csv'
    umap_df.to_csv(umap_path, index=False)
    data_paths['umap_coordinates'] = str(umap_path)
    
    # Save PCA coordinates
    if X_pca is not None:
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df = pd.concat([df[['File Name', 'chemical_type', 'concentration', 'session_id', 'source_file']].reset_index(drop=True),
                            pca_df], axis=1)
        pca_path = data_dir / 'pca_coordinates.csv'
        pca_df.to_csv(pca_path, index=False)
        data_paths['pca_coordinates'] = str(pca_path)
    
    # Save t-SNE coordinates
    if X_tsne is not None:
        tsne_df = pd.DataFrame(X_tsne, columns=['tSNE_1', 'tSNE_2'])
        tsne_df = pd.concat([df[['File Name', 'chemical_type', 'concentration', 'session_id', 'source_file']].reset_index(drop=True),
                             tsne_df], axis=1)
        tsne_path = data_dir / 'tsne_coordinates.csv'
        tsne_df.to_csv(tsne_path, index=False)
        data_paths['tsne_coordinates'] = str(tsne_path)
    
    # Save cluster labels
    cluster_df = pd.DataFrame({
        'File Name': df['File Name'].values,
        'chemical_type': df['chemical_type'].values,
        'concentration': df['concentration'].values,
        'session_id': df['session_id'].values,
        'source_file': df['source_file'].values,
        'cluster_label': cluster_labels
    })
    cluster_path = data_dir / 'cluster_labels.csv'
    cluster_df.to_csv(cluster_path, index=False)
    data_paths['cluster_labels'] = str(cluster_path)
    
    # Save raw feature array
    feature_df = pd.DataFrame(X_array, columns=feature_cols)
    feature_df = pd.concat([df[['File Name', 'chemical_type', 'concentration', 'session_id', 'source_file']].reset_index(drop=True),
                           feature_df], axis=1)
    feature_path = data_dir / 'raw_features.csv'
    feature_df.to_csv(feature_path, index=False)
    data_paths['raw_features'] = str(feature_path)
    
    # Save mean feature vectors per chemical_type/concentration combination
    if mean_vectors is not None and mean_metadata is not None:
        mean_vec_df = pd.DataFrame(mean_vectors, columns=feature_cols)
        mean_metadata_df = pd.DataFrame(mean_metadata)
        mean_vec_df = pd.concat([mean_metadata_df.reset_index(drop=True), mean_vec_df], axis=1)
        mean_vec_path = data_dir / 'mean_feature_vectors_per_group.csv'
        mean_vec_df.to_csv(mean_vec_path, index=False)
        data_paths['mean_feature_vectors'] = str(mean_vec_path)
        
        # Save mean vector coordinates
        if mean_pca is not None:
            mean_pca_df = pd.DataFrame(mean_pca, columns=['PC1', 'PC2'])
            mean_pca_df = pd.concat([mean_metadata_df.reset_index(drop=True), mean_pca_df], axis=1)
            mean_pca_path = data_dir / 'mean_vectors_pca_coordinates.csv'
            mean_pca_df.to_csv(mean_pca_path, index=False)
            data_paths['mean_pca_coordinates'] = str(mean_pca_path)
        
        if mean_umap is not None:
            mean_umap_df = pd.DataFrame(mean_umap, columns=['UMAP_1', 'UMAP_2'])
            mean_umap_df = pd.concat([mean_metadata_df.reset_index(drop=True), mean_umap_df], axis=1)
            mean_umap_path = data_dir / 'mean_vectors_umap_coordinates.csv'
            mean_umap_df.to_csv(mean_umap_path, index=False)
            data_paths['mean_umap_coordinates'] = str(mean_umap_path)
        
        if mean_tsne is not None:
            mean_tsne_df = pd.DataFrame(mean_tsne, columns=['tSNE_1', 'tSNE_2'])
            mean_tsne_df = pd.concat([mean_metadata_df.reset_index(drop=True), mean_tsne_df], axis=1)
            mean_tsne_path = data_dir / 'mean_vectors_tsne_coordinates.csv'
            mean_tsne_df.to_csv(mean_tsne_path, index=False)
            data_paths['mean_tsne_coordinates'] = str(mean_tsne_path)
    
    # Save correlation matrix
    if corr_matrix is not None:
        corr_path = data_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_path, index=True)
        data_paths['correlation_matrix'] = str(corr_path)
    
    return data_paths


def save_tables(df: pd.DataFrame, summary_stats: pd.DataFrame,
               anova_results: pd.DataFrame, pairwise_results: pd.DataFrame,
               feature_importance: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Save all analysis tables to CSV files.
    """
    tables_dir = Path(output_dir) / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    table_paths = {}
    
    # Save combined data
    data_dir = Path(output_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / 'combined_data.csv'
    df.to_csv(data_path, index=False)
    table_paths['combined_data'] = str(data_path)
    
    # Save summary statistics
    if not summary_stats.empty:
        path = tables_dir / 'summary_statistics.csv'
        summary_stats.to_csv(path, index=False)
        table_paths['summary_stats'] = str(path)
    
    # Save ANOVA results
    if not anova_results.empty:
        path = tables_dir / 'anova_results.csv'
        anova_results.to_csv(path, index=False)
        table_paths['anova'] = str(path)
    
    # Save pairwise comparisons
    if not pairwise_results.empty:
        path = tables_dir / 'pairwise_comparisons.csv'
        pairwise_results.to_csv(path, index=False)
        table_paths['pairwise'] = str(path)
    
    # Save feature importance
    if not feature_importance.empty:
        path = tables_dir / 'feature_importance.csv'
        feature_importance.to_csv(path, index=False)
        table_paths['feature_importance'] = str(path)
    
    return table_paths


def run_full_analysis(data_dir: str, output_dir: str, 
                     n_clusters: Optional[int] = None,
                     alpha: float = 0.05,
                     random_state: int = 42) -> Dict:
    """
    Run complete analysis pipeline.
    Returns dictionary with all results and file paths.
    """
    print("Loading data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df)} samples")
    
    print("Preparing features...")
    X, X_array, feature_cols = prepare_features(df)
    print(f"Prepared {len(feature_cols)} features")
    
    print("Performing dimensionality reduction (UMAP)...")
    X_umap = perform_dimensionality_reduction(X_array, method='umap', 
                                              n_components=2, random_state=random_state)
    
    print("Performing clustering...")
    cluster_results = perform_clustering(X_array, n_clusters=n_clusters, 
                                        random_state=random_state)
    print(f"Identified {cluster_results['n_clusters']} clusters")
    
    print("Calculating summary statistics...")
    summary_stats = calculate_summary_statistics(df, feature_cols, 
                                                group_col='chemical_type')
    
    print("Performing statistical tests (ANOVA)...")
    anova_results = perform_statistical_tests(df, feature_cols, 
                                             group_col='chemical_type', alpha=alpha)
    
    print("Performing pairwise comparisons...")
    pairwise_results = perform_pairwise_tests(df, feature_cols, 
                                             group_col='chemical_type', alpha=alpha)
    
    print("Calculating feature importance...")
    feature_importance = calculate_feature_importance(df, feature_cols, 
                                                    target_col='chemical_type',
                                                    random_state=random_state)
    
    print("Creating visualizations...")
    figure_paths, raw_data = create_visualizations(df, X_umap, cluster_results, 
                                                    feature_cols, output_dir, X_array=X_array)
    
    print("Saving raw data...")
    # Compute PCA and t-SNE for individual samples if not already computed
    X_pca = raw_data.get('X_pca')
    X_tsne = raw_data.get('X_tsne')
    
    if X_pca is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        raw_data['X_pca'] = X_pca
    
    if X_tsne is None and X_array is not None:
        print("Computing t-SNE for individual samples...")
        X_tsne = perform_dimensionality_reduction(X_array, method='tsne', 
                                                  n_components=2, random_state=random_state)
        raw_data['X_tsne'] = X_tsne
    
    raw_data_paths = save_raw_data(
        df=df,
        X_array=X_array,
        X_umap=X_umap,
        X_pca=raw_data.get('X_pca'),
        X_tsne=raw_data.get('X_tsne'),
        cluster_labels=cluster_results['labels'],
        feature_cols=feature_cols,
        mean_vectors=raw_data.get('mean_vectors'),
        mean_metadata=raw_data.get('mean_metadata'),
        mean_pca=raw_data.get('mean_pca'),
        mean_umap=raw_data.get('mean_umap'),
        mean_tsne=raw_data.get('mean_tsne'),
        corr_matrix=raw_data.get('corr_matrix'),
        output_dir=output_dir
    )
    
    print("Saving tables...")
    table_paths = save_tables(df, summary_stats, anova_results, 
                             pairwise_results, feature_importance, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    
    return {
        'data': df,
        'cluster_results': cluster_results,
        'summary_stats': summary_stats,
        'anova_results': anova_results,
        'pairwise_results': pairwise_results,
        'feature_importance': feature_importance,
        'figure_paths': figure_paths,
        'table_paths': table_paths,
        'raw_data_paths': raw_data_paths,
        'output_dir': output_dir
    }
