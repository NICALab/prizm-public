"""
Napari widget for PRIZM 2-stage MoA prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from napari.qt.threading import thread_worker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QTextEdit,
    QToolButton,
    QWidget,
)

from prizm_napari._widget import PandasModel
from prizm_napari.input_discovery import (
    discover_perfish_workbooks,
    infer_perfish_group_name,
    is_control_like_workbook,
)
from prizm_napari.moa_analysis import run_full_moa_analysis
from prizm_napari.workbook_selection_dialogs import (
    GroupNameEditorDialog,
    OrderedWorkbookSelectionDialog,
)

if TYPE_CHECKING:
    import napari


class PRIZMMoAPredictionQWidget(QWidget):
    """
    2-stage hierarchical MoA prediction widget.
    """

    def __init__(self, napari_viewer=None, viewer=None):
        super().__init__()
        self.viewer = viewer or napari_viewer
        self._discovered_files = []
        self._train_files = []
        self._vehicle_files = []
        self._unknown_files = []
        self._train_group_names = {}

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
        row = 0

        self.le_data_dir = QLineEdit()
        self.le_data_dir.setReadOnly(True)
        self.le_data_dir.setPlaceholderText("Select a root folder containing PerFishMetrics_*.xlsx (recursive search supported)")
        btn_data = QPushButton("Select Data Dir...", self)
        btn_data.clicked.connect(self._browse_data_dir)
        layout.addWidget(QLabel("Excel Root Directory (recursive)"), row, 0)
        layout.addWidget(self.le_data_dir, row, 1)
        layout.addWidget(btn_data, row, 2)
        row += 1

        self.lbl_discovery = QLabel("No folder selected.", self)
        layout.addWidget(self.lbl_discovery, row, 0, 1, 3)
        row += 1

        self.btn_pick_roles = QPushButton("Pick TRAIN / Vehicle / UNKNOWN...", self)
        self.btn_pick_roles.clicked.connect(self._pick_roles)
        layout.addWidget(self.btn_pick_roles, row, 0, 1, 3)
        row += 1

        self.te_selection_summary = QTextEdit(self)
        self.te_selection_summary.setReadOnly(True)
        self.te_selection_summary.setPlaceholderText("TRAIN / Vehicle / UNKNOWN selections will be listed here.")
        self.te_selection_summary.setMinimumHeight(180)
        layout.addWidget(QLabel("Selected Roles"), row, 0, 1, 3)
        row += 1
        layout.addWidget(self.te_selection_summary, row, 0, 1, 3)
        row += 1

        self.le_output_dir = QLineEdit()
        self.le_output_dir.setReadOnly(True)
        btn_output = QPushButton("Select Output Dir...", self)
        btn_output.clicked.connect(self._browse_output_dir)
        layout.addWidget(QLabel("Output Directory"), row, 0)
        layout.addWidget(self.le_output_dir, row, 1)
        layout.addWidget(btn_output, row, 2)
        row += 1

        self.cb_make_figures = QCheckBox("Generate visual reports", self)
        self.cb_make_figures.setChecked(True)
        layout.addWidget(self.cb_make_figures, row, 0, 1, 3)
        row += 1

        self.cb_include_train = QCheckBox("Include TRAIN files in prediction outputs", self)
        self.cb_include_train.setChecked(True)
        layout.addWidget(self.cb_include_train, row, 0, 1, 3)
        row += 1

        self.btn_training_params = QToolButton(self)
        self.btn_training_params.setText("Training Parameters")
        self.btn_training_params.setCheckable(True)
        self.btn_training_params.setChecked(False)
        self.btn_training_params.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_training_params.clicked.connect(self._toggle_training_params)
        layout.addWidget(self.btn_training_params, row, 0, 1, 3)
        row += 1

        self.training_params_widget = QWidget(self)
        training_layout = QGridLayout()
        training_layout.setContentsMargins(18, 0, 0, 0)
        self.training_params_widget.setLayout(training_layout)
        layout.addWidget(self.training_params_widget, row, 0, 1, 3)
        row += 1

        training_row = 0

        self.le_target_fpr = QLineEdit("0.05")
        training_layout.addWidget(QLabel("Target FPR"), training_row, 0)
        training_layout.addWidget(self.le_target_fpr, training_row, 1, 1, 2)
        training_row += 1

        self.le_file_tox_mean_thr = QLineEdit("0.90")
        training_layout.addWidget(QLabel("Min Match Fraction"), training_row, 0)
        training_layout.addWidget(self.le_file_tox_mean_thr, training_row, 1, 1, 2)
        training_row += 1

        self.le_file_tox_frac_thr = QLineEdit("S1_LOGI")
        training_layout.addWidget(QLabel("Stage1 Final ID"), training_row, 0)
        training_layout.addWidget(self.le_file_tox_frac_thr, training_row, 1, 1, 2)
        training_row += 1

        self.sb_kfold = QSpinBox()
        self.sb_kfold.setRange(2, 10)
        self.sb_kfold.setValue(5)
        training_layout.addWidget(QLabel("CV Folds"), training_row, 0)
        training_layout.addWidget(self.sb_kfold, training_row, 1, 1, 2)
        training_row += 1

        self.cb_use_robust = QCheckBox("Use robust control stats (median/MAD)", self)
        self.cb_use_robust.setChecked(True)
        training_layout.addWidget(self.cb_use_robust, training_row, 0, 1, 3)
        training_row += 1

        self.le_sim_metric = QLineEdit("euclid")
        training_layout.addWidget(QLabel("Similarity Metric"), training_row, 0)
        training_layout.addWidget(self.le_sim_metric, training_row, 1, 1, 2)
        training_row += 1

        self.sb_sim_top_k = QSpinBox()
        self.sb_sim_top_k.setRange(1, 20)
        self.sb_sim_top_k.setValue(3)
        training_layout.addWidget(QLabel("Similarity Top-K"), training_row, 0)
        training_layout.addWidget(self.sb_sim_top_k, training_row, 1, 1, 2)
        training_row += 1

        self.le_self_label = QLineEdit("Self")
        training_layout.addWidget(QLabel("Self Similarity Label"), training_row, 0)
        training_layout.addWidget(self.le_self_label, training_row, 1, 1, 2)
        training_row += 1

        self.le_tost_alpha = QLineEdit("0.05")
        training_layout.addWidget(QLabel("Dominance Alpha"), training_row, 0)
        training_layout.addWidget(self.le_tost_alpha, training_row, 1, 1, 2)
        training_row += 1

        self.le_sim_multiplier = QLineEdit("16")
        training_layout.addWidget(QLabel("Permutation Max Exact N"), training_row, 0)
        training_layout.addWidget(self.le_sim_multiplier, training_row, 1, 1, 2)
        training_row += 1

        self.le_clip_z = QLineEdit("6.0")
        training_layout.addWidget(QLabel("Clip Z"), training_row, 0)
        training_layout.addWidget(self.le_clip_z, training_row, 1, 1, 2)
        training_row += 1

        self.le_missing_frac = QLineEdit("0.3")
        training_layout.addWidget(QLabel("Max Missing Fraction"), training_row, 0)
        training_layout.addWidget(self.le_missing_frac, training_row, 1, 1, 2)
        training_row += 1

        self.sb_top_features = QSpinBox()
        self.sb_top_features.setRange(0, 500)
        self.sb_top_features.setValue(0)
        training_layout.addWidget(QLabel("Top Features (ANOVA)"), training_row, 0)
        training_layout.addWidget(self.sb_top_features, training_row, 1, 1, 2)
        training_row += 1

        self.sb_n_trees = QSpinBox()
        self.sb_n_trees.setRange(10, 1000)
        self.sb_n_trees.setValue(200)
        training_layout.addWidget(QLabel("Bagged Trees"), training_row, 0)
        training_layout.addWidget(self.sb_n_trees, training_row, 1, 1, 2)
        training_row += 1

        self.sb_rng_seed = QSpinBox()
        self.sb_rng_seed.setRange(0, 99999)
        self.sb_rng_seed.setValue(0)
        training_layout.addWidget(QLabel("Random Seed"), training_row, 0)
        training_layout.addWidget(self.sb_rng_seed, training_row, 1, 1, 2)
        training_row += 1

        self.cb_outlier_train = QCheckBox("Include self in similarity", self)
        self.cb_outlier_train.setChecked(True)
        training_layout.addWidget(self.cb_outlier_train, training_row, 0, 1, 3)
        training_row += 1

        self.cb_outlier_unknown = QCheckBox("Save dominance statistics", self)
        self.cb_outlier_unknown.setChecked(True)
        training_layout.addWidget(self.cb_outlier_unknown, training_row, 0, 1, 3)
        training_row += 1

        self.le_outlier_top_percent = QLineEdit("mean")
        training_layout.addWidget(QLabel("Dominance Competitor Mode"), training_row, 0)
        training_layout.addWidget(self.le_outlier_top_percent, training_row, 1, 1, 2)
        training_row += 1

        self.le_outlier_min_weight = QLineEdit("10000")
        training_layout.addWidget(QLabel("Permutation N"), training_row, 0)
        training_layout.addWidget(self.le_outlier_min_weight, training_row, 1, 1, 2)
        training_row += 1

        self.btn_run = QPushButton("Run 2-Stage MoA", self)
        self.btn_run.clicked.connect(self._start_analysis)
        layout.addWidget(self.btn_run, row, 0, 1, 3)
        row += 1

        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        layout.addWidget(self.pbar, row, 0, 1, 3)

        self._current_worker = None
        self._toggle_training_params(False)
        self._update_selection_summary()

    def _toggle_training_params(self, checked=None):
        if checked is None:
            checked = self.btn_training_params.isChecked()
        checked = bool(checked)
        self.btn_training_params.setChecked(checked)
        self.training_params_widget.setVisible(checked)
        self.btn_training_params.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Excel Root Directory", "")
        if d:
            self.le_data_dir.setText(d)
            self._train_files = []
            self._vehicle_files = []
            self._unknown_files = []
            self._train_group_names = {}
            self._refresh_discovered_files()

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if d:
            self.le_output_dir.setText(d)

    def _refresh_discovered_files(self):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            self._discovered_files = []
            self.lbl_discovery.setText("No folder selected.")
            self._update_selection_summary()
            return
        self._discovered_files = discover_perfish_workbooks(data_dir, recursive=True)
        self.lbl_discovery.setText(
            f"Discovered {len(self._discovered_files)} PerFishMetrics workbook(s) under the selected root."
        )
        self._update_selection_summary()

    def _pick_roles(self):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            QMessageBox.warning(self, "MoA 2-Stage", "Select a valid root directory first.")
            return

        self._refresh_discovered_files()
        if not self._discovered_files:
            QMessageBox.warning(self, "MoA 2-Stage", "No PerFishMetrics workbooks were found in that directory.")
            return

        train_files = OrderedWorkbookSelectionDialog.get_selected(
            self._discovered_files,
            title="Pick TRAIN Files",
            prompt="Select TRAIN files (include control / vehicle files) and arrange them in the desired order.",
            root=data_dir,
            initial_checked=self._train_files,
            parent=self,
        )
        if train_files is None:
            return
        if not train_files:
            QMessageBox.warning(self, "MoA 2-Stage", "Select at least one TRAIN workbook.")
            return

        initial_vehicle = [fp for fp in self._vehicle_files if fp in train_files]
        if not initial_vehicle:
            initial_vehicle = [fp for fp in train_files if is_control_like_workbook(fp)]
        if not initial_vehicle:
            initial_vehicle = train_files[:1]

        vehicle_files = OrderedWorkbookSelectionDialog.get_selected(
            train_files,
            title="Pick Vehicle(Control)",
            prompt="Select the Vehicle(Control) workbook(s) from the chosen TRAIN set.",
            root=data_dir,
            initial_checked=initial_vehicle,
            parent=self,
        )
        if vehicle_files is None:
            return
        if not vehicle_files:
            QMessageBox.warning(self, "MoA 2-Stage", "Select at least one Vehicle(Control) workbook.")
            return

        non_vehicle = [fp for fp in train_files if fp not in vehicle_files]
        group_map = {}
        if non_vehicle:
            initial_groups = {
                str(Path(fp).resolve()): self._train_group_names.get(str(Path(fp).resolve()), infer_perfish_group_name(fp))
                for fp in non_vehicle
            }
            group_map = GroupNameEditorDialog.get_group_map(
                non_vehicle,
                title="Edit Non-Vehicle TRAIN Group Names",
                root=data_dir,
                initial_groups=initial_groups,
                parent=self,
            )
            if group_map is None:
                return

        remaining = [fp for fp in self._discovered_files if fp not in train_files]
        if remaining:
            unknown_files = OrderedWorkbookSelectionDialog.get_selected(
                remaining,
                title="Pick UNKNOWN Files",
                prompt="Select UNKNOWN files for prediction.",
                root=data_dir,
                initial_checked=[fp for fp in self._unknown_files if fp in remaining],
                parent=self,
            )
            if unknown_files is None:
                return
        else:
            unknown_files = []

        self._train_files = [Path(fp).resolve() for fp in train_files]
        self._vehicle_files = [Path(fp).resolve() for fp in vehicle_files]
        self._unknown_files = [Path(fp).resolve() for fp in unknown_files]
        self._train_group_names = group_map
        self._update_selection_summary()

    def _update_selection_summary(self):
        data_dir = self.le_data_dir.text().strip()
        root = Path(data_dir) if data_dir else None
        if not self._train_files:
            self.te_selection_summary.setPlainText("No TRAIN / Vehicle / UNKNOWN roles selected yet.")
            return

        lines = [
            f"TRAIN files: {len(self._train_files)}",
            f"Vehicle files: {len(self._vehicle_files)}",
            f"UNKNOWN files: {len(self._unknown_files)}",
            "",
            "[TRAIN]",
        ]
        vehicle_set = {str(fp) for fp in self._vehicle_files}
        for fp in self._train_files:
            try:
                rel = fp.relative_to(root) if root is not None else fp.name
            except Exception:
                rel = fp
            if str(fp) in vehicle_set:
                group = "Vehicle"
            else:
                group = self._train_group_names.get(str(fp), infer_perfish_group_name(fp))
            lines.append(f"- {group} | {rel.as_posix()}")

        lines.append("")
        lines.append("[UNKNOWN]")
        if self._unknown_files:
            for fp in self._unknown_files:
                try:
                    rel = fp.relative_to(root) if root is not None else fp.name
                except Exception:
                    rel = fp
                lines.append(f"- {infer_perfish_group_name(fp)} | {rel.as_posix()}")
        else:
            lines.append("- None selected")
        self.te_selection_summary.setPlainText("\n".join(lines))

    def _build_train_spec(self) -> pd.DataFrame:
        data_dir = self.le_data_dir.text().strip()
        root = Path(data_dir).resolve()
        vehicle_set = {str(fp) for fp in self._vehicle_files}
        rows = []
        for fp in self._train_files:
            fp = Path(fp).resolve()
            try:
                rel = fp.relative_to(root).as_posix()
            except Exception:
                rel = fp.name
            if str(fp) in vehicle_set:
                group = "Vehicle"
            else:
                group = self._train_group_names.get(str(fp), infer_perfish_group_name(fp))
            rows.append({"Group": group, "File": rel, "FullPath": str(fp)})
        spec = pd.DataFrame(rows)
        veh = spec["Group"] == "Vehicle"
        return pd.concat([spec[veh], spec[~veh]], ignore_index=True)

    @thread_worker
    def _analysis_thread(self):
        data_dir = self.le_data_dir.text().strip()
        out_dir = self.le_output_dir.text().strip() or None
        train_spec = self._build_train_spec()
        unknown_files = list(self._unknown_files)

        try:
            target_fpr = float(self.le_target_fpr.text())
        except ValueError:
            target_fpr = 0.05
        try:
            min_match_frac = float(self.le_file_tox_mean_thr.text())
        except ValueError:
            min_match_frac = 0.90
        stage1_final_id = self.le_file_tox_frac_thr.text().strip() or "S1_LOGI"
        try:
            clip_z_value = float(self.le_clip_z.text())
        except ValueError:
            clip_z_value = 6.0
        try:
            missing_frac_max = float(self.le_missing_frac.text())
        except ValueError:
            missing_frac_max = 0.3
        dominance_competitor_mode = self.le_outlier_top_percent.text().strip().lower() or "mean"
        if dominance_competitor_mode not in {"mean", "top2mean", "best"}:
            dominance_competitor_mode = "mean"
        try:
            perm_n = int(float(self.le_outlier_min_weight.text()))
        except ValueError:
            perm_n = 10000
        try:
            dominance_alpha = float(self.le_tost_alpha.text())
        except ValueError:
            dominance_alpha = 0.05
        try:
            perm_max_exact_n = int(float(self.le_sim_multiplier.text()))
        except ValueError:
            perm_max_exact_n = 16
        sim_metric = self.le_sim_metric.text().strip() or "euclid"
        self_label = self.le_self_label.text().strip() or "Self"

        yield (
            0,
            1,
            f"Running MoA analysis with {len(train_spec)} TRAIN, {int((train_spec['Group'] == 'Vehicle').sum())} Vehicle, and {len(unknown_files)} UNKNOWN workbook(s)...",
        )
        result = run_full_moa_analysis(
            train_dir=data_dir,
            unknown_dir=data_dir,
            out_dir=out_dir,
            train_spec=train_spec,
            unknown_files=unknown_files,
            use_robust_control_stats=self.cb_use_robust.isChecked(),
            clip_z_value=clip_z_value,
            missing_frac_max=missing_frac_max,
            n_top_features=self.sb_top_features.value(),
            kfold_fish=self.sb_kfold.value(),
            rng_seed=self.sb_rng_seed.value(),
            n_trees=self.sb_n_trees.value(),
            target_fpr=target_fpr,
            make_figures=self.cb_make_figures.isChecked(),
            stage1_final_id=stage1_final_id,
            sim_metric=sim_metric,
            sim_top_k=self.sb_sim_top_k.value(),
            include_self_in_similarity=self.cb_outlier_train.isChecked(),
            self_similarity_label=self_label,
            save_dominance_stats=self.cb_outlier_unknown.isChecked(),
            dominance_alpha=dominance_alpha,
            exclude_self_in_dominance=True,
            save_dominance_stats_ml=self.cb_outlier_unknown.isChecked(),
            dominance_competitor_mode=dominance_competitor_mode,
            perm_n=perm_n,
            perm_max_exact_n=perm_max_exact_n,
            perm_seed=self.sb_rng_seed.value(),
            save_tost_vs_self=self.cb_outlier_unknown.isChecked(),
            tost_alpha=dominance_alpha,
            tost_delta_softmax=0.15,
            tost_delta_distance=1.18,
            sim_multiplier=1.5,
            include_train_in_analysis=self.cb_include_train.isChecked(),
            min_match_frac=min_match_frac,
        )
        yield (1, 1, "MoA analysis complete")
        return result

    def _start_analysis(self):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            QMessageBox.warning(self, "MoA 2-Stage", "Select a valid root directory first.")
            return
        if not self._train_files or not self._vehicle_files:
            QMessageBox.warning(
                self,
                "MoA 2-Stage",
                "Pick TRAIN / Vehicle / UNKNOWN roles first. At least one Vehicle(Control) workbook is required.",
            )
            return

        self.btn_run.setEnabled(False)
        self.pbar.setMaximum(100)
        self.pbar.setValue(0)
        self.pbar.setFormat("Starting...")

        worker = self._analysis_thread()
        self._current_worker = worker
        worker.yielded.connect(self._on_progress_update)
        worker.returned.connect(self._on_analysis_complete)
        worker.errored.connect(self._on_analysis_error)
        worker.start()

    def _on_progress_update(self, info):
        if isinstance(info, tuple) and len(info) == 3:
            cur, total, msg = info
            if total > 0:
                pct = int((cur / total) * 100)
                self.pbar.setValue(pct)
            self.pbar.setFormat(msg)

    def _on_analysis_error(self, error):
        self.btn_run.setEnabled(True)
        self.pbar.setFormat("Error")
        self._current_worker = None
        try:
            QMessageBox.critical(self, "MoA 2-Stage Error", str(error))
        except Exception:
            pass

    def _on_analysis_complete(self, result):
        self.btn_run.setEnabled(True)
        self.pbar.setValue(100)
        self.pbar.setFormat("Complete")
        self._current_worker = None

        if not result:
            return

        summary_df = pd.DataFrame(
            [
                {
                    "Output Directory": result.get("out_dir"),
                    "Bundle": result.get("bundle_path"),
                    "Train Report": result.get("train_report_xlsx"),
                    "Master": result.get("master_xlsx"),
                    "Unknown Files": result.get("n_unknown_files"),
                }
            ]
        )
        view = QTableView()
        model = PandasModel(summary_df)
        view.setModel(model)
        view.resizeColumnsToContents()
        if self.viewer is not None:
            self.viewer.window.add_dock_widget(
                view,
                name="MoA 2-Stage Summary",
                area="bottom",
            )
