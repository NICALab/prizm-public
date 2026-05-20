from typing import TYPE_CHECKING

import os
import re
import numpy as np
import tifffile
import dask
import pandas as pd
import skimage.io as skio
from datetime import datetime

import napari
import napari.layers
from napari.qt.threading import thread_worker
from PyQt5.QtCore import QAbstractTableModel, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
    QTableView,
    QTabWidget,
    QCheckBox,
    QTextEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

try:
    import onnx
except Exception:  # pragma: no cover - optional dependency
    onnx = None

from prizm_napari.infer import PRIZMInference
from prizm_napari.analysis import compute_segmentation_statistics, compute_functional_statistics, compute_synchronize_analysis, combine_results
from prizm_napari.utils import overlay_time_series
from prizm_napari.batch_segmentation_core import run_batch_segmentation_core, _discover_sample_dirs

if TYPE_CHECKING:
    import napari


DEFAULT_BATCH_ROOT_DIR = ""
DEFAULT_BATCH_OUTPUT_DIR = ""
DEFAULT_BATCH_MODEL_PATH = ""
DEFAULT_BATCH_METADATA_MODE = "Manual Entry"
DEFAULT_BATCH_RESIZE_SCALE = "0.9210"
DEFAULT_BATCH_FRAME_INTERVAL = "0.062"

class PandasModel(QAbstractTableModel):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return str(self._df.iat[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])


# class PRIZMSegmentationQWidget(QWidget):
#     def __init__(self, viewer: "napari.viewer.Viewer"):
#         super().__init__()
#         self.viewer = viewer
#         self.masks = None

#         # Layout
#         grid_layout = QGridLayout()
#         grid_layout.setAlignment(Qt.AlignTop)
#         self.setLayout(grid_layout)

#         # Image selector
#         self.cb_image = QComboBox()
#         self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#         grid_layout.addWidget(QLabel("Image", self), 0, 0)
#         grid_layout.addWidget(self.cb_image, 0, 1, 1, 2)

#         # Channel selector
#         self.le_channel = QLineEdit()
#         self.le_channel.setPlaceholderText("e.g. 0 (red or gray), 1 (green), 2 (blue)")
#         grid_layout.addWidget(QLabel("Channel to segment", self), 1, 0)
#         grid_layout.addWidget(self.le_channel, 1, 1, 1, 2)

#         # Output directory selector
#         self.le_output_dir = QLineEdit()
#         self.le_output_dir.setReadOnly(True)
#         btn_output = QPushButton("Select Output Dir...", self)
#         btn_output.clicked.connect(self._browse_output_dir)
#         grid_layout.addWidget(QLabel("Output Directory", self), 2, 0)
#         grid_layout.addWidget(self.le_output_dir, 2, 1)
#         grid_layout.addWidget(btn_output, 2, 2)

#         # Model file path and browse button
#         self.le_model_path = QLineEdit()
#         self.le_model_path.setReadOnly(True)
#         btn_browse = QPushButton("Browse Model...", self)
#         btn_browse.clicked.connect(self._browse_model_file)
#         grid_layout.addWidget(QLabel("Model File", self), 3, 0)
#         grid_layout.addWidget(self.le_model_path, 3, 1)
#         grid_layout.addWidget(btn_browse, 3, 2)

#         # Parameters...
#         self.le_backbone = QLineEdit()
#         self.le_backbone.setPlaceholderText("e.g. resnet50")
#         grid_layout.addWidget(QLabel("Backbone", self), 4, 0)
#         grid_layout.addWidget(self.le_backbone, 4, 1, 1, 2)

#         self.sb_encoder_depth = QSpinBox()
#         self.sb_encoder_depth.setRange(1, 10)
#         self.sb_encoder_depth.setValue(5)
#         grid_layout.addWidget(QLabel("Encoder Depth", self), 5, 0)
#         grid_layout.addWidget(self.sb_encoder_depth, 5, 1, 1, 2)

#         self.sb_decoder_channels = QSpinBox()
#         self.sb_decoder_channels.setRange(1, 1024)
#         self.sb_decoder_channels.setValue(256)
#         grid_layout.addWidget(QLabel("Decoder Channels", self), 6, 0)
#         grid_layout.addWidget(self.sb_decoder_channels, 6, 1, 1, 2)

#         self.sb_encoder_output_stride = QSpinBox()
#         self.sb_encoder_output_stride.setRange(1, 32)
#         self.sb_encoder_output_stride.setValue(16)
#         grid_layout.addWidget(QLabel("Encoder Output Stride", self), 7, 0)
#         grid_layout.addWidget(self.sb_encoder_output_stride, 7, 1, 1, 2)

#         self.le_atrous_rates = QLineEdit()
#         self.le_atrous_rates.setPlaceholderText("e.g. 6 12 18")
#         grid_layout.addWidget(QLabel("Atrous Rates", self), 8, 0)
#         grid_layout.addWidget(self.le_atrous_rates, 8, 1, 1, 2)

#         # Compute button
#         self.btn = QPushButton("Run Segmentation", self)
#         self.btn.clicked.connect(self._start_inference)
#         grid_layout.addWidget(self.btn, 9, 0, 1, 3)

#         # Metadata mode selector
#         self.cb_meta_mode = QComboBox()
#         self.cb_meta_mode.addItems(["Use Metadata XML", "Manual Entry"])
#         self.cb_meta_mode.currentIndexChanged.connect(self._on_meta_mode_changed)
#         grid_layout.addWidget(QLabel("Metadata Mode", self), 10, 0)
#         grid_layout.addWidget(self.cb_meta_mode, 10, 1, 1, 2)

#         # Metadata file loader
#         self.le_metadata_path = QLineEdit()
#         self.le_metadata_path.setReadOnly(True)
#         btn_metadata = QPushButton("Browse Metadata...", self)
#         btn_metadata.clicked.connect(self._browse_metadata_file)
#         grid_layout.addWidget(QLabel("Metadata File", self), 11, 0)
#         grid_layout.addWidget(self.le_metadata_path, 11, 1)
#         grid_layout.addWidget(btn_metadata, 11, 2)
#         self.btn_metadata = btn_metadata

#         # Manual metadata inputs
#         self.le_resize_scale = QLineEdit()
#         self.le_resize_scale.setPlaceholderText("e.g. 0.5")
#         grid_layout.addWidget(QLabel("Resize Scale"), 12, 0)
#         grid_layout.addWidget(self.le_resize_scale, 12, 1, 1, 2)

#         self.le_relative_interval = QLineEdit()
#         self.le_relative_interval.setPlaceholderText("e.g. 0.062")
#         grid_layout.addWidget(QLabel("Relative Time Interval (sec)"), 13, 0)
#         grid_layout.addWidget(self.le_relative_interval, 13, 1, 1, 2)

#         # Analysis button
#         self.btn_analysis = QPushButton("Run Analysis", self)
#         self.btn_analysis.setEnabled(False)
#         self.btn_analysis.clicked.connect(self._start_analysis)
#         grid_layout.addWidget(self.btn_analysis, 14, 0, 1, 3)

#         # Progress bar
#         self.pbar = QProgressBar(self, minimum=0, maximum=1)
#         grid_layout.addWidget(self.pbar, 15, 0, 1, 3)

#         # Layer callbacks
#         self.viewer.layers.events.inserted.connect(lambda e: e.value.events.name.connect(self._on_layer_change))
#         self.viewer.layers.events.inserted.connect(self._on_layer_change)
#         self.viewer.layers.events.removed.connect(self._on_layer_change)
#         self._on_layer_change(None)
#         self._on_meta_mode_changed()
    
#     def _on_meta_mode_changed(self):
#         self.meta_manual = self.cb_meta_mode.currentText() == "Manual Entry"
#         self.le_metadata_path.setEnabled(not self.meta_manual)
#         self.btn_metadata.setEnabled(not self.meta_manual)
#         self.le_resize_scale.setEnabled(self.meta_manual)
#         self.le_relative_interval.setEnabled(self.meta_manual)
    
#     def _browse_model_file(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self,
#             "Select Model File",
#             "",
#             "PyTorch Model Files (*.pth);;All Files *)",
#         )
#         if path:
#             self.le_model_path.setText(path)

#     def _browse_metadata_file(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self,
#             "Select Metadata File",
#             "",
#             "XML Files (*.xml);;All Files *)",
#         )
#         if path:
#             self.le_metadata_path.setText(path)

#     def _browse_output_dir(self):
#         directory = QFileDialog.getExistingDirectory(
#             self,
#             "Select Output Directory",
#             "",
#         )
#         if directory:
#             self.le_output_dir.setText(directory)

#     def _on_layer_change(self, e):
#         self.cb_image.clear()
#         for layer in self.viewer.layers:
#             if isinstance(layer, napari.layers.Image):
#                 self.cb_image.addItem(layer.name, layer.data)
#                 print(self.cb_image.currentText())

#     @thread_worker
#     def _infer_thread(self):
#         return self.infer.infer(self.selected_image, self.selected_segmentation_channel)

#     def _start_inference(self):
#         model_path = self.le_model_path.text()
#         if not model_path:
#             return
#         backbone = self.le_backbone.text() or "resnet50"
#         encoder_depth = self.sb_encoder_depth.value()
#         decoder_channels = self.sb_decoder_channels.value()
#         encoder_output_stride = self.sb_encoder_output_stride.value()
#         try:
#             atrous_rates = [int(r) for r in self.le_atrous_rates.text().split()]
#         except ValueError:
#             atrous_rates = []

#         self.infer = PRIZMInference(
#             model_path,
#             num_classes=3,
#             backbone=backbone,
#             encoder_depth=encoder_depth,
#             decoder_channels=decoder_channels,
#             encoder_output_stride=encoder_output_stride,
#             decoder_atrous_rates=atrous_rates,
#         )

#         self.selected_image = self.cb_image.currentData()
#         if self.selected_image is None:
#             return
        
#         self.selected_segmentation_channel = int(self.le_channel.text())
#         if not self.selected_segmentation_channel:
#             return

#         # disable metadata until segmentation done
#         self.le_metadata_path.setEnabled(False)
#         self.btn_metadata.setEnabled(False)

#         self.pbar.setMaximum(0)
#         worker = self._infer_thread()
#         worker.returned.connect(self._on_segmented)
#         worker.start()

#     def _on_segmented(self, masks):
#         if masks is not None:
#             # display segmentation
#             layer = self.viewer.add_labels(masks, name=f"{self.cb_image.currentText()} CH{self.selected_segmentation_channel} Segmentation")
#             layer.opacity = 0.5
#             layer.blending = "additive"
#             self.masks = masks

#             # save segmentation mask
#             out_dir = self.le_output_dir.text()
#             if out_dir:
#                 mask_path = os.path.join(out_dir, f'{self.cb_image.currentText()}_ch{self.selected_segmentation_channel}_segmentation.tif')
#                 tifffile.imwrite(mask_path, masks.astype(np.uint8))

#             # enable analysis and metadata loader
#             self.btn_analysis.setEnabled(True)
#             self.le_metadata_path.setEnabled(True)
#             self.btn_metadata.setEnabled(True)
#         self.pbar.setMaximum(1)
        
#     @thread_worker
#     def _analysis_thread(self):
        
#         if self.meta_manual:
#             meta_file = None
#             meta_info = {
#                 "resize_scale": float(self.le_resize_scale.text()) if self.le_resize_scale.text() else 1.0,
#                 "frame_interval": float(self.le_relative_interval.text()) if self.le_relative_interval.text() else 0.062,
#             }
#         else:
#             meta_file = self.le_metadata_path.text()
#             meta_info = None
#         stats_df, fig_v_axis, fig_a_axis, viz_data = compute_segmentation_statistics(self.masks, f"{self.cb_image.currentText()}_ch{self.selected_segmentation_channel}", self.le_output_dir.text(), meta_file=meta_file, meta_info=meta_info)
        
#         # Compute the functional statistics
#         v_peaks_df, vFS_df, a_peaks_df, fig_v, fig_vfs, fig_a, fig_va, viz_data = compute_functional_statistics(stats_df, f"{self.cb_image.currentText()}_ch{self.selected_segmentation_channel}", self.le_output_dir.text(), viz_data=viz_data)
        
#         # Compute synchronize analysis
#         sync_df, fig_cav, fig_cc, viz_data = compute_synchronize_analysis(stats_df, f"{self.cb_image.currentText()}_ch{self.selected_segmentation_channel}", self.le_output_dir.text(), viz_data=viz_data)
        
#         # Combine the results
#         combined_df, viz_data = combine_results(f"{self.cb_image.currentText()}_ch{self.selected_segmentation_channel}", stats_df, v_peaks_df, vFS_df, a_peaks_df, sync_df, self.le_output_dir.text(), viz_data=viz_data)
        
#         return (stats_df, v_peaks_df, vFS_df, a_peaks_df, combined_df, fig_v_axis, fig_a_axis, fig_v, fig_vfs, fig_a, fig_va, fig_cav, fig_cc, viz_data)

#     def _start_analysis(self):
                
#         if not self.le_output_dir.text():
#             return
#         if self.masks is None:
#             return
#         if len(self.masks.squeeze()) == 3:
#             return
#         if self.meta_manual:
#             if not self.le_resize_scale.text() or not self.le_relative_interval.text():
#                 return
#         else:
#             if not self.le_metadata_path.text():
#                 return
        
#         # disable button, show progress
#         self.btn_analysis.setEnabled(False)
        
#         self.pbar.setMaximum(0)
#         worker = self._analysis_thread()
#         worker.returned.connect(self._on_analyzed)
#         worker.start()
        
#     def _on_analyzed(self, dfs):
#         stats_df, v_peaks_df, vFS_df, a_peaks_df, combined_df, fig_v_axis, fig_a_axis, fig_v, fig_vfs, fig_a, fig_va, fig_cav, fig_cc, viz_data = dfs

#         # ——— Figures as another tab widget ———
#         figs = [
#             ("Ventricle peaks", fig_v),
#             ("Atrium peaks",   fig_a),
#             ("Ventricle + Atrium", fig_va),
#             ("Ventricle axis", fig_v_axis),
#             ("Atrium axis",    fig_a_axis),
#             ("Major axis peaks", fig_vfs),
#             ("Cavity signal",  fig_cav),
#             ("Cross-correlation",     fig_cc),
#         ]
#         tab_plots = QTabWidget()
#         for name, fig in figs:
#             canvas = FigureCanvas(fig)
#             tab_plots.addTab(canvas, name)
#         self.viewer.window.add_dock_widget(
#             tab_plots,
#             name="Analysis Plots",
#             area="bottom"
#         )
        
#         # ——— Tables as one tab widget ———
#         tables = [
#             ("Combined stats", combined_df),
#             ("Segmentation stats", stats_df),
#             ("Ventricle stats", v_peaks_df),
#             ("Atrium stats", a_peaks_df),
#             ("Fractional shortening stats", vFS_df),
#         ]
#         tab_tables = QTabWidget()
#         for name, df in tables:
#             view = QTableView()
#             view.setModel(PandasModel(df))
#             view.resizeColumnsToContents()
#             view.resizeRowsToContents()
#             tab_tables.addTab(view, name)
#         self.viewer.window.add_dock_widget(
#             tab_tables,
#             name="Analysis Statistics",
#             area="right"
#         )

#         # ——— Enhanced visualization overlay ———
#         if hasattr(self, 'selected_image') and self.selected_image is not None:
#             # Create enhanced overlay with V-A relationships
#             overlay_ts = overlay_time_series(
#                 self.selected_image, 
#                 self.masks, 
#                 stats_df, 
#                 viz_data=viz_data,
#                 output_dir=None, 
#                 mask_alpha=0.2
#             )
            
#             # Add overlay to viewer
#             self.viewer.add_image(
#                 overlay_ts,
#                 name=f"{self.cb_image.currentText()} CH{self.selected_segmentation_channel} Analysis Overlay",
#                 blending="additive",
#             )

#         # ——— restore UI ———
#         self.pbar.setMaximum(1)
#         self.btn_analysis.setEnabled(True)
        
        
class PRIZMBatchSegmentationQWidget(QWidget):
    """
    Batch segmentation and full analysis over a root directory with organized structure.
    Expected directory structure:
        {CHEMICAL_TYPE}_{CONCENTRATION}/
            {UNIQUE_SAMPLE_DIR_NAME}/
                frame files (.png/.tif/.jpg/.jpeg)
            [optional metadata dirs/files in sample or chemical folder]

    Output CSV files are saved with format compatible with chemical analysis plugin:
        {DATE}_{CHEMICAL_TYPE}_{CONCENTRATION}_{SAMPLE_ID}.csv
    
    All individual combine_results DataFrames are aggregated into a single batch DataFrame,
    which is saved once at the end and displayed in Napari.
    """
    def __init__(self, napari_viewer=None, viewer=None):
        super().__init__()
        self.viewer = viewer or napari_viewer

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Root data directory selector
        self.le_root_dir = QLineEdit()
        self.le_root_dir.setReadOnly(True)
        self.le_root_dir.setText(DEFAULT_BATCH_ROOT_DIR)
        self.le_root_dir.setPlaceholderText("Select the root directory containing PRIZM raw data")
        btn_root = QPushButton("Select Data Dir...", self)
        btn_root.clicked.connect(self._browse_root_dir)
        layout.addWidget(QLabel("Root Data Directory"), 0, 0)
        layout.addWidget(self.le_root_dir, 0, 1)
        layout.addWidget(btn_root, 0, 2)

        # Metadata mode selector (XML vs. manual)
        self.cb_meta_mode = QComboBox()
        self.cb_meta_mode.addItems(["Use Metadata XML", "Manual Entry"])
        self.cb_meta_mode.setCurrentText(DEFAULT_BATCH_METADATA_MODE)
        self.cb_meta_mode.currentIndexChanged.connect(self._on_meta_mode_changed)
        layout.addWidget(QLabel("Metadata Mode"), 1, 0)
        layout.addWidget(self.cb_meta_mode, 1, 1, 1, 2)

        # Manual metadata inputs (for manual mode)
        self.le_resize_scale = QLineEdit()
        self.le_resize_scale.setPlaceholderText("e.g. 0.9210")
        self.le_resize_scale.setText(DEFAULT_BATCH_RESIZE_SCALE)
        layout.addWidget(QLabel("Resize Scale"), 2, 0)
        layout.addWidget(self.le_resize_scale, 2, 1, 1, 2)

        self.le_relative_interval = QLineEdit()
        self.le_relative_interval.setPlaceholderText("e.g. 0.062")
        self.le_relative_interval.setText(DEFAULT_BATCH_FRAME_INTERVAL)
        layout.addWidget(QLabel("Relative Time Interval"), 3, 0)
        layout.addWidget(self.le_relative_interval, 3, 1, 1, 2)

        # Output directory selector
        self.le_out_dir = QLineEdit()
        self.le_out_dir.setReadOnly(True)
        self.le_out_dir.setText(DEFAULT_BATCH_OUTPUT_DIR)
        self.le_out_dir.setPlaceholderText("Select where batch segmentation outputs should be saved")
        btn_out = QPushButton("Select Output Dir...", self)
        btn_out.clicked.connect(self._browse_out_dir)
        layout.addWidget(QLabel("Output Directory"), 4, 0)
        layout.addWidget(self.le_out_dir, 4, 1)
        layout.addWidget(btn_out, 4, 2)

        # Model file selector
        self.le_model_path = QLineEdit()
        self.le_model_path.setReadOnly(True)
        self.le_model_path.setText(DEFAULT_BATCH_MODEL_PATH)
        self.le_model_path.setPlaceholderText("Browse to a .onnx or .pth segmentation model")
        btn_model = QPushButton("Browse Model...", self)
        btn_model.clicked.connect(self._browse_model_file)
        layout.addWidget(QLabel("Model File"), 5, 0)
        layout.addWidget(self.le_model_path, 5, 1)
        layout.addWidget(btn_model, 5, 2)

        self.cb_model_type = QComboBox()
        self.cb_model_type.addItems(["onnx", "pth", "auto"])
        self.cb_model_type.setCurrentText("onnx")
        self.cb_model_type.currentIndexChanged.connect(self._on_model_type_changed)
        layout.addWidget(QLabel("Model Type"), 6, 0)
        layout.addWidget(self.cb_model_type, 6, 1, 1, 2)

        # Image mode selector (channel vs. grayscale)
        self.cb_image_mode = QComboBox()
        self.cb_image_mode.addItems(["Select Channel", "Convert to Grayscale"])
        self.cb_image_mode.currentIndexChanged.connect(self._on_image_mode_changed)
        layout.addWidget(QLabel("Channel Mode"), 7, 0)
        layout.addWidget(self.cb_image_mode, 7, 1, 1, 2)
        self.grayscale = False

        # Channel input
        self.le_channel = QLineEdit()
        self.le_channel.setPlaceholderText("e.g. 1 (green), 0 (red/gray), 2 (blue)")
        self.le_channel.setText("1")
        layout.addWidget(QLabel("Channel to segment"), 8, 0)
        layout.addWidget(self.le_channel, 8, 1, 1, 2)

        # Backbone
        self.le_backbone = QLineEdit()
        self.le_backbone.setPlaceholderText("e.g. resnet50")
        self.le_backbone.setText("resnet50")
        layout.addWidget(QLabel("Backbone"), 9, 0)
        layout.addWidget(self.le_backbone, 9, 1, 1, 2)

        # Encoder depth
        self.sb_enc_depth = QSpinBox()
        self.sb_enc_depth.setRange(1, 10)
        self.sb_enc_depth.setValue(5)
        layout.addWidget(QLabel("Encoder Depth"), 10, 0)
        layout.addWidget(self.sb_enc_depth, 10, 1, 1, 2)

        # Decoder channels
        self.sb_dec_ch = QSpinBox()
        self.sb_dec_ch.setRange(1, 1024)
        self.sb_dec_ch.setValue(256)
        layout.addWidget(QLabel("Decoder Channels"), 11, 0)
        layout.addWidget(self.sb_dec_ch, 11, 1, 1, 2)

        # Encoder output stride
        self.sb_out_stride = QSpinBox()
        self.sb_out_stride.setRange(1, 32)
        self.sb_out_stride.setValue(16)
        layout.addWidget(QLabel("Encoder Output Stride"), 12, 0)
        layout.addWidget(self.sb_out_stride, 12, 1, 1, 2)

        # Atrous rates
        self.le_atrous = QLineEdit()
        self.le_atrous.setPlaceholderText("e.g. 3 6 9")
        self.le_atrous.setText("6 12 18")
        layout.addWidget(QLabel("Atrous Rates"), 13, 0)
        layout.addWidget(self.le_atrous, 13, 1, 1, 2)

        # Model input channels
        self.sb_input_channels = QSpinBox()
        self.sb_input_channels.setRange(1, 4)
        self.sb_input_channels.setValue(3)
        layout.addWidget(QLabel("Model Input Channels"), 14, 0)
        layout.addWidget(self.sb_input_channels, 14, 1, 1, 2)

        # Inference batch size
        self.sb_infer_batch = QSpinBox()
        self.sb_infer_batch.setRange(1, 4096)
        self.sb_infer_batch.setValue(1)
        layout.addWidget(QLabel("Inference Batch Size"), 15, 0)
        layout.addWidget(self.sb_infer_batch, 15, 1, 1, 2)

        # Postprocess masks checkbox
        self.cb_postprocess_masks = QCheckBox(
            "Postprocess masks before saving and analysis", self
        )
        self.cb_postprocess_masks.setChecked(False)
        self.cb_postprocess_masks.stateChanged.connect(
            self._on_postprocess_masks_checkbox_changed
        )
        layout.addWidget(self.cb_postprocess_masks, 16, 0, 1, 3)
        self.infer_postprocess = False

        # ——— Load results into viewer checkbox ———
        self.cb_load = QCheckBox("Load images and segmentations to napari", self)
        self.cb_load.setChecked(False)
        self.cb_load.stateChanged.connect(self._on_load_results_checkbox_changed)
        layout.addWidget(self.cb_load, 17, 0, 1, 3)
        self.load_to_viewer = False

        # ——— Save analysis visulzation overlay checkbox ———
        self.cb_analysis_vis = QCheckBox("Generate analysis visualization overlay", self)
        self.cb_analysis_vis.setChecked(False)
        self.cb_analysis_vis.stateChanged.connect(self._on_save_analysis_vis_checkbox_changed)
        layout.addWidget(self.cb_analysis_vis, 18, 0, 1, 3)
        self.save_analysis_vis = False

        # Run/Stop button
        self.btn_run = QPushButton("Run Batch", self)
        self.btn_run.clicked.connect(self._start_batch)
        layout.addWidget(self.btn_run, 19, 0, 1, 3)
        
        # Stop button (initially hidden)
        self.btn_stop = QPushButton("Stop", self)
        self.btn_stop.clicked.connect(self._stop_batch)
        self.btn_stop.setVisible(False)
        layout.addWidget(self.btn_stop, 19, 0, 1, 3)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.pbar, 20, 0, 1, 3)

        # Live batch log
        self.te_batch_log = QTextEdit(self)
        self.te_batch_log.setReadOnly(True)
        self.te_batch_log.setPlaceholderText("Batch run log will appear here.")
        self.te_batch_log.setMinimumHeight(180)
        layout.addWidget(QLabel("Batch Log"), 21, 0, 1, 3)
        layout.addWidget(self.te_batch_log, 22, 0, 1, 3)
        
        # Store worker reference and cancellation flag
        self._current_worker = None
        self._cancel_requested = False
        self._batch_log_path = None
        self._last_progress_message = None

        # Initialize metadata-mode UI
        self._on_meta_mode_changed()
        self._sync_model_backend_ui()
    
    def _on_load_results_checkbox_changed(self, state):
        self.load_to_viewer = (state == Qt.Checked)
        
    def _on_save_analysis_vis_checkbox_changed(self, state):
        self.save_analysis_vis = (state == Qt.Checked)

    def _on_postprocess_masks_checkbox_changed(self, state):
        self.infer_postprocess = (state == Qt.Checked)

    def _on_image_mode_changed(self):
        self.grayscale = (self.cb_image_mode.currentText() == "Convert to Grayscale")
        self.le_channel.setEnabled(not self.grayscale)

    def _on_meta_mode_changed(self):
        self.meta_manual = (self.cb_meta_mode.currentText() == "Manual Entry")
        # Manual mode widgets
        self.le_resize_scale.setEnabled(self.meta_manual)
        self.le_relative_interval.setEnabled(self.meta_manual)

    def _is_effective_onnx_backend(self) -> bool:
        model_type = self.cb_model_type.currentText().strip().lower()
        if model_type == "onnx":
            return True
        if model_type == "pth":
            return False
        model_path = self.le_model_path.text().strip().lower()
        return model_path.endswith(".onnx")

    def _detect_onnx_input_channels(self, model_path: str):
        if onnx is None:
            return None
        if not model_path or not os.path.isfile(model_path):
            return None
        if not str(model_path).lower().endswith(".onnx"):
            return None
        try:
            model = onnx.load(str(model_path))
            if not model.graph.input:
                return None
            input_tensor = model.graph.input[0].type.tensor_type
            dims = input_tensor.shape.dim
            if len(dims) < 2:
                return None
            channels = int(dims[1].dim_value)
            if channels > 0:
                return channels
        except Exception:
            return None
        return None

    def _sync_model_backend_ui(self):
        using_onnx = self._is_effective_onnx_backend()
        arch_widgets = [
            self.le_backbone,
            self.sb_enc_depth,
            self.sb_dec_ch,
            self.sb_out_stride,
            self.le_atrous,
        ]
        arch_tooltip = (
            "Ignored for ONNX models; the exported ONNX graph already fixes the network architecture."
            if using_onnx
            else ""
        )
        for widget in arch_widgets:
            widget.setEnabled(not using_onnx)
            widget.setToolTip(arch_tooltip)

        if using_onnx:
            detected_channels = self._detect_onnx_input_channels(self.le_model_path.text().strip())
            if detected_channels is not None:
                self.sb_input_channels.setValue(int(detected_channels))
                self.sb_input_channels.setEnabled(False)
                self.sb_input_channels.setToolTip(
                    f"Detected from ONNX input shape: {int(detected_channels)} channel(s)."
                )
            else:
                self.sb_input_channels.setEnabled(True)
                self.sb_input_channels.setToolTip(
                    "For ONNX models this should match the ONNX input channel count."
                )
        else:
            self.sb_input_channels.setEnabled(True)
            self.sb_input_channels.setToolTip("")

    def _on_model_type_changed(self, *_args):
        self._sync_model_backend_ui()
    
    def _browse_root_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root Data Directory", "")
        if d:
            self.le_root_dir.setText(d)

    def _browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if d:
            self.le_out_dir.setText(d)

    def _browse_model_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.onnx *.pth);;ONNX Model Files (*.onnx);;PyTorch Model Files (*.pth);;All Files (*)",
        )
        if f:
            self.le_model_path.setText(f)
            suffix = os.path.splitext(f)[1].lower()
            if suffix == ".onnx":
                self.cb_model_type.setCurrentText("onnx")
            elif suffix == ".pth":
                self.cb_model_type.setCurrentText("pth")
            self._sync_model_backend_ui()

    @thread_worker
    def _batch_thread(self):
        root_dir = self.le_root_dir.text()
        out_dir = self.le_out_dir.text()
        
        if self.grayscale:
            channel = 0
        else:
            try:
                channel = int(self.le_channel.text())
            except ValueError:
                return

        backbone = self.le_backbone.text() or "resnet50"
        model_type = self.cb_model_type.currentText()
        encoder_depth = self.sb_enc_depth.value()
        decoder_channels = self.sb_dec_ch.value()
        encoder_output_stride = self.sb_out_stride.value()
        input_channels = self.sb_input_channels.value()
        infer_batch_size = self.sb_infer_batch.value()
        try:
            atrous_rates = [int(r) for r in self.le_atrous.text().split()]
        except ValueError:
            atrous_rates = []

        resize_scale = None
        frame_interval = None
        if self.meta_manual:
            try:
                resize_scale = float(self.le_resize_scale.text())
                frame_interval = float(self.le_relative_interval.text())
            except ValueError:
                pass

        # Count total samples first for progress tracking
        import os
        chem_conc_dirs = sorted(
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        )
        total_samples = 0
        for chem_conc_dir in chem_conc_dirs:
            chem_conc_path = os.path.join(root_dir, chem_conc_dir)
            sample_dirs = _discover_sample_dirs(chem_conc_path)
            total_samples += len(sample_dirs)
        
        if total_samples == 0:
            yield (0, 1, "No samples found")
            return
        
        # Progress tracking using a shared variable that can be updated from callback
        import threading
        from queue import Queue
        
        progress_queue = Queue()
        result_container = [None]
        exception_container = [None]
        
        def progress_callback(event):
            if isinstance(event, (int, float)):
                progress_queue.put(('progress', int(event)))
            else:
                progress_queue.put(('log', str(event)))
        
        def run_core():
            try:
                result_container[0] = run_batch_segmentation_core(
                    root_dir=root_dir,
                    out_dir=out_dir,
                    model_path=self.le_model_path.text(),
                    model_type=model_type,
                    channel=channel,
                    grayscale=self.grayscale,
                    backbone=backbone,
                    encoder_depth=encoder_depth,
                    decoder_channels=decoder_channels,
                    encoder_output_stride=encoder_output_stride,
                    atrous_rates=atrous_rates,
                    input_channels=input_channels,
                    meta_manual=self.meta_manual,
                    resize_scale=resize_scale,
                    frame_interval=frame_interval,
                    metadata_file=None,
                    load_to_viewer=self.load_to_viewer,
                    save_analysis_vis=self.save_analysis_vis,
                    infer_postprocess=self.infer_postprocess,
                    infer_batch_size=infer_batch_size,
                    progress_callback=progress_callback,
                )
                progress_queue.put(('done', None))
            except Exception as e:
                exception_container[0] = e
                progress_queue.put(('error', None))
        
        yield (0, total_samples, f"Starting batch segmentation ({total_samples} samples)...")
        
        # Start core function in a thread
        core_thread = threading.Thread(target=run_core, daemon=True)
        core_thread.start()
        
        # Monitor progress and yield updates
        last_progress = 0
        while core_thread.is_alive() or not progress_queue.empty():
            # Check for cancel request
            if self._cancel_requested:
                progress_queue.put(('abort', None))
                yield (last_progress, total_samples, "Cancelling...")
                # Try to stop the core thread (it will finish current sample)
                break
            
            try:
                # Check for progress updates (non-blocking)
                while not progress_queue.empty():
                    msg_type, value = progress_queue.get_nowait()
                    if msg_type == 'progress':
                        current = value
                        if current > last_progress:
                            last_progress = current
                            yield (current, total_samples, f"Processing sample {current}/{total_samples}")
                    elif msg_type == 'log':
                        yield {"log": value}
                    elif msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        break
                    elif msg_type == 'abort':
                        break
            except:
                pass
            
            # Small sleep to avoid busy waiting
            import time
            time.sleep(0.05)
        
        # Wait for thread to finish (with timeout)
        core_thread.join(timeout=2.0)
        
        # Check if cancelled
        if self._cancel_requested:
            yield (last_progress, total_samples, "Cancelled")
            return None
        
        if exception_container[0]:
            raise exception_container[0]
        
        yield (total_samples, total_samples, "Batch segmentation complete!")
        return result_container[0]

    def _start_batch(self):
        if (
            not self.le_root_dir.text()
            or not self.le_out_dir.text()
            or not self.le_model_path.text()
        ):
            return
        
        if not self.grayscale:
            try:
                int(self.le_channel.text())
            except ValueError:
                return
        
        if self.meta_manual:
            if not self.le_resize_scale.text() or not self.le_relative_interval.text():
                return
        # In XML mode, no manual entries are required; metadata is auto-loaded per video

        os.makedirs(self.le_out_dir.text(), exist_ok=True)
        self.te_batch_log.clear()
        self._last_progress_message = None
        run_ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._batch_log_path = os.path.join(
            self.le_out_dir.text(),
            f"batch_segmentation_gui_log_{run_ts}.txt",
        )

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
        self._append_batch_log("Starting batch segmentation run")
        self._append_batch_log(f"Root Data Directory: {self.le_root_dir.text()}")
        self._append_batch_log(f"Output Directory: {self.le_out_dir.text()}")
        self._append_batch_log(f"Model File: {self.le_model_path.text()}")
        self._append_batch_log(
            f"Model Type: {self.cb_model_type.currentText()} | "
            f"Channel Mode: {self.cb_image_mode.currentText()} | "
            f"Channel: {'grayscale' if self.grayscale else self.le_channel.text()} | "
            f"Input Channels: {self.sb_input_channels.value()} | "
            f"Inference Batch Size: {self.sb_infer_batch.value()}"
        )
        if self.meta_manual:
            self._append_batch_log(
                f"Metadata Mode: Manual Entry | Resize Scale: {self.le_resize_scale.text()} | "
                f"Relative Time Interval: {self.le_relative_interval.text()}"
            )
        else:
            self._append_batch_log("Metadata Mode: Use Metadata XML")
        
        worker = self._batch_thread()
        self._current_worker = worker
        worker.returned.connect(self._on_batch_complete)
        worker.yielded.connect(self._on_progress_update)
        worker.errored.connect(self._on_batch_error)
        worker.start()
    
    def _stop_batch(self):
        """Stop the current batch processing"""
        self._cancel_requested = True
        self.btn_stop.setEnabled(False)
        self.pbar.setFormat("Cancelling...")

    def _finalize_batch_ui(self, status_text: str, *, completed: bool = False):
        self.pbar.setMaximum(100)
        if completed:
            self.pbar.setValue(100)
        else:
            self.pbar.setValue(self.pbar.value())
        self.pbar.setFormat(status_text)
        self.btn_run.setEnabled(True)
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self.btn_stop.setEnabled(False)
        self._current_worker = None

    def _append_batch_log(self, message: str):
        text = str(message or "").rstrip()
        if not text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        self.te_batch_log.append(line)
        if self._batch_log_path:
            try:
                with open(self._batch_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
    
    def _on_batch_error(self, error):
        """Handle errors from batch processing"""
        message = str(error).strip() if error is not None else "Batch segmentation failed."
        short_message = message.splitlines()[-1] if message else "Batch segmentation failed."
        self._append_batch_log(f"ERROR: {short_message}")
        self._finalize_batch_ui(f"Error: {short_message}")
        try:
            QMessageBox.critical(self, "Batch Segmentation Error", message)
        except Exception:
            pass
    
    def _on_progress_update(self, progress_info):
        """Update progress bar from worker thread"""
        import time
        
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()

        if isinstance(progress_info, dict):
            log_message = progress_info.get("log")
            if log_message:
                self._append_batch_log(log_message)
            current = progress_info.get("current")
            total = progress_info.get("total")
            message = progress_info.get("message")
            if current is None or total is None or message is None:
                return
            progress_info = (current, total, message)
        
        if isinstance(progress_info, tuple) and len(progress_info) == 3:
            current, total, message = progress_info
            if message != self._last_progress_message:
                self._append_batch_log(message)
                self._last_progress_message = message
            if total > 0:
                percent = int((current / total) * 100)
                self.pbar.setValue(percent)
                
                # Calculate timing information
                elapsed = time.time() - self._start_time
                if current > 0:
                    rate = current / elapsed  # samples per second
                    remaining_samples = total - current
                    if rate > 0:
                        remaining_time = remaining_samples / rate
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

    def _on_batch_complete(self, res):
        # finalize UI
        if res is None:
            self._finalize_batch_ui("Cancelled")
            return

        self._finalize_batch_ui("Complete!", completed=True)
        batch_combined_df, img_list, analysis_vis_overlay = res

        # Display batch combined table in Napari
        if not batch_combined_df.empty:
            view = QTableView()
            model = PandasModel(batch_combined_df)
            view.setModel(model)
            view.resizeColumnsToContents()
            self.viewer.window.add_dock_widget(
                view,
                name="Combined Analysis Statistics",
                area="bottom"
            )
        
        # optionally load results into Napari
        if self.load_to_viewer:
            for video_name, stack, masks, analysis_vis_overlay in img_list:
                
                # load the raw stack as an Image layer
                self.viewer.add_image(
                    stack.compute() if hasattr(stack, "compute") else np.array(stack),
                    name=f"{video_name} Original",
                )
                # load the segmentation as a Labels layer
                self.viewer.add_labels(
                    masks,
                    name=f"{video_name} Segmentation",
                )
                if self.save_analysis_vis:
                    # load the analysis overlay as an Image layer
                    self.viewer.add_image(
                        analysis_vis_overlay,
                        name=f"{video_name} Analysis Overlay",
                        blending="additive",
                    )
            
