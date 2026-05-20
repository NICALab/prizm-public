"""
Napari widget for chemical analysis of functional feature vectors.
"""

from typing import TYPE_CHECKING
import os
from pathlib import Path

import napari
from napari.qt.threading import thread_worker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
    QTableView,
    QTabWidget,
    QCheckBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from prizm_napari._widget import PandasModel

if TYPE_CHECKING:
    import napari


class PRIZMChemicalAnalysisQWidget(QWidget):
    """
    Widget for analyzing chemical analysis data from PRIZM video analysis.
    Loads CSV files with functional feature vectors and performs comprehensive
    statistical and visual analysis based on chemical type and concentration labels.
    """
    
    def __init__(self, napari_viewer=None, viewer=None):
        super().__init__()
        self.viewer = viewer or napari_viewer
        
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        
        row = 0
        
        # Data directory selector
        self.le_data_dir = QLineEdit()
        self.le_data_dir.setReadOnly(True)
        btn_data = QPushButton("Select Data Dir...", self)
        btn_data.clicked.connect(self._browse_data_dir)
        layout.addWidget(QLabel("Data Directory"), row, 0)
        layout.addWidget(self.le_data_dir, row, 1)
        layout.addWidget(btn_data, row, 2)
        row += 1
        
        # Output directory selector
        self.le_output_dir = QLineEdit()
        self.le_output_dir.setReadOnly(True)
        btn_output = QPushButton("Select Output Dir...", self)
        btn_output.clicked.connect(self._browse_output_dir)
        layout.addWidget(QLabel("Output Directory"), row, 0)
        layout.addWidget(self.le_output_dir, row, 1)
        layout.addWidget(btn_output, row, 2)
        row += 1
        
        # Visualization toggle
        self.cb_visualize = QCheckBox("Display results in napari", self)
        self.cb_visualize.setChecked(True)
        layout.addWidget(self.cb_visualize, row, 0, 1, 3)
        row += 1
        
        # Number of clusters
        self.sb_n_clusters = QSpinBox()
        self.sb_n_clusters.setRange(2, 20)
        self.sb_n_clusters.setValue(0)  # 0 means auto
        self.sb_n_clusters.setSpecialValueText("Auto")
        layout.addWidget(QLabel("Number of Clusters"), row, 0)
        layout.addWidget(self.sb_n_clusters, row, 1, 1, 2)
        row += 1
        
        # Significance level
        self.le_alpha = QLineEdit()
        self.le_alpha.setText("0.05")
        self.le_alpha.setPlaceholderText("e.g. 0.05")
        layout.addWidget(QLabel("Significance Level (α)"), row, 0)
        layout.addWidget(self.le_alpha, row, 1, 1, 2)
        row += 1
        
        # Run/Stop button
        self.btn_run = QPushButton("Run Analysis", self)
        self.btn_run.clicked.connect(self._start_analysis)
        layout.addWidget(self.btn_run, row, 0, 1, 3)
        
        # Stop button (initially hidden)
        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.clicked.connect(self._stop_analysis)
        self.btn_stop.setVisible(False)
        layout.addWidget(self.btn_stop, row, 0, 1, 3)
        row += 1
        
        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.pbar, row, 0, 1, 3)
        
        # Store results, worker reference, and cancellation flag
        self.results = None
        self._current_worker = None
        self._cancel_requested = False
    
    def _browse_data_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            "",
        )
        if directory:
            self.le_data_dir.setText(directory)
    
    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
        )
        if directory:
            self.le_output_dir.setText(directory)
    
    @thread_worker
    def _analysis_thread(self):
        # Lazy import to avoid loading heavy dependencies at module import time
        from prizm_napari.chemical_analysis import (
            load_data, prepare_features, perform_dimensionality_reduction,
            perform_clustering, calculate_summary_statistics, perform_statistical_tests,
            perform_pairwise_tests, calculate_feature_importance, create_visualizations,
            save_raw_data, save_tables
        )
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        data_dir = self.le_data_dir.text()
        output_dir = self.le_output_dir.text()
        
        n_clusters = self.sb_n_clusters.value()
        if n_clusters == 0:
            n_clusters = None
        
        try:
            alpha = float(self.le_alpha.text())
        except ValueError:
            alpha = 0.05
        
        # Define progress steps
        total_steps = 10
        current_step = 0
        
        yield (current_step, total_steps, "Loading data...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        df = load_data(data_dir)
        current_step += 1
        
        yield (current_step, total_steps, "Preparing features...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        X, X_array, feature_cols = prepare_features(df)
        current_step += 1
        
        yield (current_step, total_steps, "Performing dimensionality reduction (UMAP)...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        X_umap = perform_dimensionality_reduction(X_array, method='umap', 
                                                  n_components=2, random_state=42)
        current_step += 1
        
        yield (current_step, total_steps, "Performing clustering...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        cluster_results = perform_clustering(X_array, n_clusters=n_clusters, 
                                            random_state=42)
        current_step += 1
        
        yield (current_step, total_steps, "Calculating summary statistics...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        summary_stats = calculate_summary_statistics(df, feature_cols, 
                                                    group_col='chemical_type')
        current_step += 1
        
        yield (current_step, total_steps, "Performing statistical tests (ANOVA)...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        anova_results = perform_statistical_tests(df, feature_cols, 
                                                 group_col='chemical_type', alpha=alpha)
        current_step += 1
        
        yield (current_step, total_steps, "Performing pairwise comparisons...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        pairwise_results = perform_pairwise_tests(df, feature_cols, 
                                                 group_col='chemical_type', alpha=alpha)
        current_step += 1
        
        yield (current_step, total_steps, "Calculating feature importance...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        feature_importance = calculate_feature_importance(df, feature_cols, 
                                                        target_col='chemical_type',
                                                        random_state=42)
        current_step += 1
        
        yield (current_step, total_steps, "Creating visualizations...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        figure_paths, raw_data = create_visualizations(df, X_umap, cluster_results, 
                                                        feature_cols, output_dir, X_array=X_array)
        current_step += 1
        
        yield (current_step, total_steps, "Saving data...")
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        # Compute PCA and t-SNE if needed
        X_pca = raw_data.get('X_pca')
        X_tsne = raw_data.get('X_tsne')
        
        if X_pca is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            raw_data['X_pca'] = X_pca
        
        if X_tsne is None and X_array is not None:
            X_tsne = perform_dimensionality_reduction(X_array, method='tsne', 
                                                      n_components=2, random_state=42)
            raw_data['X_tsne'] = X_tsne
        
        raw_data_paths = save_raw_data(
            df=df, X_array=X_array, X_umap=X_umap,
            X_pca=raw_data.get('X_pca'), X_tsne=raw_data.get('X_tsne'),
            cluster_labels=cluster_results['labels'], feature_cols=feature_cols,
            mean_vectors=raw_data.get('mean_vectors'),
            mean_metadata=raw_data.get('mean_metadata'),
            mean_pca=raw_data.get('mean_pca'),
            mean_umap=raw_data.get('mean_umap'),
            mean_tsne=raw_data.get('mean_tsne'),
            corr_matrix=raw_data.get('corr_matrix'),
            output_dir=output_dir
        )
        
        table_paths = save_tables(df, summary_stats, anova_results, 
                                 pairwise_results, feature_importance, output_dir)
        current_step += 1
        
        if self._cancel_requested:
            yield (current_step, total_steps, "Cancelled")
            return None
        
        yield (total_steps, total_steps, "Analysis complete!")
        
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
    
    def _start_analysis(self):
        if not self.le_data_dir.text():
            return
        if not self.le_output_dir.text():
            return
        
        # Validate paths
        data_dir = Path(self.le_data_dir.text())
        if not data_dir.exists():
            return
        
        output_dir = Path(self.le_output_dir.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Disable button, show progress
        self.btn_run.setEnabled(False)
        self.btn_run.setVisible(False)
        self.btn_stop.setVisible(True)
        self.btn_stop.setEnabled(True)
        self.pbar.setMaximum(100)
        self.pbar.setValue(0)
        # Reset cancellation flag and start time
        self._cancel_requested = False
        import time
        self._start_time = time.time()
        
        worker = self._analysis_thread()
        self._current_worker = worker
        worker.returned.connect(self._on_analysis_complete)
        worker.yielded.connect(self._on_progress_update)
        worker.errored.connect(self._on_analysis_error)
        worker.start()
    
    def _stop_analysis(self):
        """Stop the current analysis"""
        self._cancel_requested = True
        self.btn_stop.setEnabled(False)
        self.pbar.setFormat("Cancelling...")
    
    def _on_analysis_error(self, error):
        """Handle errors from analysis"""
        self._on_analysis_complete(None)
    
    def _on_progress_update(self, progress_info):
        """Update progress bar from worker thread"""
        import time
        
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        
        if isinstance(progress_info, tuple) and len(progress_info) == 3:
            current, total, message = progress_info
            if total > 0:
                percent = int((current / total) * 100)
                self.pbar.setValue(percent)
                
                # Calculate timing information
                elapsed = time.time() - self._start_time
                if current > 0:
                    rate = current / elapsed  # steps per second
                    remaining_steps = total - current
                    if rate > 0:
                        remaining_time = remaining_steps / rate
                        # Format time nicely
                        elapsed_str = self._format_time(elapsed)
                        remaining_str = self._format_time(remaining_time)
                        self.pbar.setFormat(f"{message} ({current}/{total}) | {elapsed_str} | {remaining_str} remaining")
                    else:
                        elapsed_str = self._format_time(elapsed)
                        self.pbar.setFormat(f"{message} ({current}/{total}) | {elapsed_str} elapsed")
                else:
                    elapsed_str = self._format_time(elapsed)
                    self.pbar.setFormat(f"{message} ({current}/{total}) | {elapsed_str} elapsed")
        elif isinstance(progress_info, (int, float)):
            self.pbar.setValue(int(progress_info))
    
    def _format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    def _on_analysis_complete(self, results):
        self.results = results
        
        # Restore UI
        self.pbar.setMaximum(100)
        if results is None:
            self.pbar.setValue(self.pbar.value())  # Keep current value
            self.pbar.setFormat("Cancelled")
        else:
            self.pbar.setValue(100)
            self.pbar.setFormat("Complete!")
        
        self.btn_run.setEnabled(True)
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._current_worker = None
        
        print(f"DEBUG: Analysis complete. Results keys: {list(results.keys()) if results else 'None'}")
        print(f"DEBUG: Visualization checkbox checked: {self.cb_visualize.isChecked()}")
        
        # Display results in napari if enabled
        if self.cb_visualize.isChecked():
            print("DEBUG: Calling _display_results...")
            self._display_results(results)
        else:
            print("DEBUG: Visualization is disabled, skipping display")
    
    def _display_results(self, results):
        """Display analysis results in napari viewer."""
        
        # Create tabbed widget for figures
        tab_plots = QTabWidget()
        
        # Load and display figures
        figure_paths = results.get('figure_paths', {})
        print(f"DEBUG: Found {len(figure_paths)} figure paths in results")
        
        if not figure_paths:
            print("DEBUG: No figure_paths found in results dictionary")
            print(f"DEBUG: Results keys: {list(results.keys())}")
            return
        
        for fig_name, fig_path in figure_paths.items():
            print(f"DEBUG: Processing figure {fig_name}: {fig_path}")
            if not os.path.exists(fig_path):
                print(f"DEBUG: Figure file does not exist: {fig_path}")
                continue
                
            try:
                from matplotlib.image import imread
                import matplotlib.pyplot as plt
                
                fig = Figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                img = imread(fig_path)
                ax.imshow(img)
                ax.axis('off')
                # No title - tab name is sufficient
                
                canvas = FigureCanvas(fig)
                tab_name = Path(fig_path).stem.replace('_', ' ').title()
                tab_plots.addTab(canvas, tab_name)
                print(f"DEBUG: Added tab '{tab_name}' for figure {fig_name}")
            except Exception as e:
                import traceback
                print(f"Error loading figure {fig_path}: {e}")
                traceback.print_exc()
        
        if tab_plots.count() > 0:
            print(f"DEBUG: Adding dock widget with {tab_plots.count()} tabs")
            self.viewer.window.add_dock_widget(
                tab_plots,
                name="Analysis Figures",
                area="bottom"
            )
            print("DEBUG: Dock widget added successfully")
        else:
            print("DEBUG: No tabs to display")
        
        # Create tabbed widget for tables
        tab_tables = QTabWidget()
        
        # Display summary statistics
        if 'summary_stats' in results and not results['summary_stats'].empty:
            view = QTableView()
            view.setModel(PandasModel(results['summary_stats']))
            view.resizeColumnsToContents()
            tab_tables.addTab(view, "Summary Statistics")
        
        # Display ANOVA results
        if 'anova_results' in results and not results['anova_results'].empty:
            view = QTableView()
            view.setModel(PandasModel(results['anova_results']))
            view.resizeColumnsToContents()
            tab_tables.addTab(view, "ANOVA Results")
        
        # Display pairwise comparisons
        if 'pairwise_results' in results and not results['pairwise_results'].empty:
            view = QTableView()
            view.setModel(PandasModel(results['pairwise_results']))
            view.resizeColumnsToContents()
            tab_tables.addTab(view, "Pairwise Comparisons")
        
        # Display feature importance
        if 'feature_importance' in results and not results['feature_importance'].empty:
            view = QTableView()
            view.setModel(PandasModel(results['feature_importance']))
            view.resizeColumnsToContents()
            tab_tables.addTab(view, "Feature Importance")
        
        # Display combined data
        if 'data' in results:
            view = QTableView()
            view.setModel(PandasModel(results['data']))
            view.resizeColumnsToContents()
            tab_tables.addTab(view, "Combined Data")
        
        if tab_tables.count() > 0:
            self.viewer.window.add_dock_widget(
                tab_tables,
                name="Analysis Tables",
                area="right"
            )
