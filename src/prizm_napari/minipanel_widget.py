"""
Napari widget for PRIZM mini-panel + heatmap + LDA/PCA/t-SNE analysis.
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
    QWidget,
)

from prizm_napari._widget import PandasModel
from prizm_napari.input_discovery import (
    discover_perfish_workbooks,
    infer_perfish_group_name,
    is_control_like_workbook,
)
from prizm_napari.minipanel_analysis import run_minipanel_analysis
from prizm_napari.workbook_selection_dialogs import MiniPanelReferenceDialog, OrderedWorkbookSelectionDialog

if TYPE_CHECKING:
    import napari


class PRIZMMiniPanelQWidget(QWidget):
    """
    Mini-panel + heatmap + LDA/PCA/t-SNE widget.
    """

    def __init__(self, napari_viewer=None, viewer=None):
        super().__init__()
        self.viewer = viewer or napari_viewer
        self._discovered_files = []
        self._selected_files = []
        self._control_group_name = ""
        self._reference_group_name = ""
        self._include_reference_in_heatmap = True
        self._save_all_pairs_excel = False

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

        self.btn_pick_files = QPushButton("Pick Files / Order...", self)
        self.btn_pick_files.clicked.connect(self._pick_files)
        layout.addWidget(self.btn_pick_files, row, 0, 1, 3)
        row += 1

        self.te_selection_summary = QTextEdit(self)
        self.te_selection_summary.setReadOnly(True)
        self.te_selection_summary.setPlaceholderText("Selected MiniPanel workbooks will be listed here.")
        self.te_selection_summary.setMinimumHeight(130)
        layout.addWidget(QLabel("Selected Workbooks"), row, 0, 1, 3)
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

        self.btn_pick_reference = QPushButton("Pick Control / Reference / Stats...", self)
        self.btn_pick_reference.clicked.connect(self._pick_reference_and_options)
        layout.addWidget(self.btn_pick_reference, row, 0, 1, 3)
        row += 1

        self.lbl_reference_summary = QLabel("No control/reference choices configured yet.", self)
        self.lbl_reference_summary.setWordWrap(True)
        layout.addWidget(self.lbl_reference_summary, row, 0, 1, 3)
        row += 1

        self.cb_make_heatmap = QCheckBox("Generate heatmap", self)
        self.cb_make_heatmap.setChecked(True)
        layout.addWidget(self.cb_make_heatmap, row, 0, 1, 3)
        row += 1

        self.cb_do_lda = QCheckBox("Run Fisher LDA", self)
        self.cb_do_lda.setChecked(True)
        layout.addWidget(self.cb_do_lda, row, 0, 1, 3)
        row += 1

        self.cb_do_pca = QCheckBox("Run PCA", self)
        self.cb_do_pca.setChecked(True)
        layout.addWidget(self.cb_do_pca, row, 0, 1, 3)
        row += 1

        self.cb_do_tsne = QCheckBox("Run t-SNE", self)
        self.cb_do_tsne.setChecked(True)
        layout.addWidget(self.cb_do_tsne, row, 0, 1, 3)
        row += 1

        self.sb_ncols = QSpinBox()
        self.sb_ncols.setRange(1, 20)
        self.sb_ncols.setValue(5)
        layout.addWidget(QLabel("Bar Panel Columns"), row, 0)
        layout.addWidget(self.sb_ncols, row, 1, 1, 2)
        row += 1

        self.btn_run = QPushButton("Run MiniPanel Analysis", self)
        self.btn_run.clicked.connect(self._start_analysis)
        layout.addWidget(self.btn_run, row, 0, 1, 3)
        row += 1

        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        layout.addWidget(self.pbar, row, 0, 1, 3)

        self._current_worker = None
        self._update_reference_summary()

    def _browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Excel Root Directory", "")
        if d:
            self.le_data_dir.setText(d)
            self._selected_files = []
            self._refresh_discovered_files(auto_select=True)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if d:
            self.le_output_dir.setText(d)

    def _refresh_discovered_files(self, auto_select: bool = False):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            self._discovered_files = []
            self.lbl_discovery.setText("No folder selected.")
            self.te_selection_summary.clear()
            self._reset_reference_selection()
            return

        self._discovered_files = discover_perfish_workbooks(data_dir, recursive=True)
        self.lbl_discovery.setText(
            f"Discovered {len(self._discovered_files)} PerFishMetrics workbook(s) under the selected root."
        )
        if auto_select and self._discovered_files:
            self._set_selected_files(self._discovered_files)
        else:
            self._update_selection_summary()

    def _pick_files(self):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            QMessageBox.warning(self, "MiniPanel", "Select a valid root directory first.")
            return

        self._refresh_discovered_files(auto_select=False)
        if not self._discovered_files:
            QMessageBox.warning(self, "MiniPanel", "No PerFishMetrics workbooks were found in that directory.")
            return

        initial_checked = self._selected_files or self._discovered_files
        selected = OrderedWorkbookSelectionDialog.get_selected(
            self._discovered_files,
            title="Pick MiniPanel Files",
            prompt="Select the workbooks to analyze and arrange them in the desired display order.",
            root=data_dir,
            initial_checked=initial_checked,
            parent=self,
        )
        if selected is None:
            return
        if not selected:
            QMessageBox.warning(self, "MiniPanel", "Select at least one workbook.")
            return
        self._set_selected_files(selected, prompt_reference=True)

    def _set_selected_files(self, files, *, prompt_reference: bool = False):
        self._selected_files = [Path(fp).resolve() for fp in files]
        self._update_selection_summary()
        self._apply_reference_defaults()
        if prompt_reference:
            self._pick_reference_and_options()

    def _update_selection_summary(self):
        data_dir = self.le_data_dir.text().strip()
        root = Path(data_dir) if data_dir else None
        if not self._selected_files:
            self.te_selection_summary.setPlainText("No workbooks selected yet.")
            return
        lines = [f"Selected {len(self._selected_files)} workbook(s):"]
        for idx, fp in enumerate(self._selected_files, start=1):
            try:
                rel = fp.relative_to(root) if root is not None else fp.name
            except Exception:
                rel = fp
            lines.append(f"{idx}. {infer_perfish_group_name(fp)} | {rel.as_posix()}")
        self.te_selection_summary.setPlainText("\n".join(lines))

    def _reset_reference_selection(self):
        self._control_group_name = ""
        self._reference_group_name = ""
        self._include_reference_in_heatmap = True
        self._save_all_pairs_excel = False
        self._update_reference_summary()

    def _apply_reference_defaults(self):
        groups = [infer_perfish_group_name(fp) for fp in self._selected_files]
        if not groups:
            self._reset_reference_selection()
            return

        default_idx = 0
        for idx, fp in enumerate(self._selected_files):
            if is_control_like_workbook(fp):
                default_idx = idx
                break

        default_group = groups[default_idx]
        if self._control_group_name not in groups:
            self._control_group_name = default_group
        if self._reference_group_name not in groups:
            self._reference_group_name = self._control_group_name
        self._update_reference_summary()

    def _pick_reference_and_options(self):
        if not self._selected_files:
            QMessageBox.warning(self, "MiniPanel", "Select MiniPanel workbooks first.")
            return

        groups = [infer_perfish_group_name(fp) for fp in self._selected_files]
        self._apply_reference_defaults()
        selection = MiniPanelReferenceDialog.get_selection(
            groups,
            default_control=self._control_group_name,
            default_reference=self._reference_group_name,
            include_reference_in_heatmap=self._include_reference_in_heatmap,
            save_all_pairs_excel=self._save_all_pairs_excel,
            parent=self,
        )
        if selection is None:
            self._update_reference_summary()
            return

        self._control_group_name = selection["control_group_name"] or self._control_group_name
        self._reference_group_name = selection["reference_group_name"] or self._control_group_name
        self._include_reference_in_heatmap = bool(selection["include_reference_in_heatmap"])
        self._save_all_pairs_excel = bool(selection["save_all_pairs_excel"])
        self._update_reference_summary()

    def _update_reference_summary(self):
        if not self._selected_files or not self._control_group_name:
            self.lbl_reference_summary.setText("No control/reference choices configured yet.")
            return

        include_text = "Yes" if self._include_reference_in_heatmap else "No"
        pairwise_text = "Yes" if self._save_all_pairs_excel else "No"
        self.lbl_reference_summary.setText(
            "\n".join(
                [
                    f"Control Group: {self._control_group_name}",
                    f"Reference Group: {self._reference_group_name or self._control_group_name}",
                    f"Include reference group in heatmap: {include_text}",
                    f"Save all pairwise Welch t-tests: {pairwise_text}",
                ]
            )
        )

    @thread_worker
    def _analysis_thread(self):
        data_dir = self.le_data_dir.text().strip()
        output_dir = self.le_output_dir.text().strip() or None
        selected_files = self._selected_files or discover_perfish_workbooks(data_dir, recursive=True)
        if not selected_files:
            raise ValueError("No PerFishMetrics workbooks selected for MiniPanel analysis.")

        self._apply_reference_defaults()
        control_name = self._control_group_name or "CTRL"
        reference_name = self._reference_group_name or control_name

        yield (0, 1, f"Running MiniPanel analysis on {len(selected_files)} selected workbook(s)...")
        result = run_minipanel_analysis(
            data_folder=data_dir,
            output_dir=output_dir,
            selected_files=selected_files,
            ordered_files=None,
            include_ctrl_in_heatmap=self._include_reference_in_heatmap,
            make_heatmap=self.cb_make_heatmap.isChecked(),
            do_lda=self.cb_do_lda.isChecked(),
            do_pca=self.cb_do_pca.isChecked(),
            do_tsne=self.cb_do_tsne.isChecked(),
            n_cols=self.sb_ncols.value(),
            control_group_name=control_name,
            reference_group_name=reference_name,
            save_all_pairs_excel=self._save_all_pairs_excel,
        )
        yield (1, 1, "MiniPanel analysis complete")
        return result

    def _start_analysis(self):
        data_dir = self.le_data_dir.text().strip()
        if not data_dir or not Path(data_dir).is_dir():
            QMessageBox.warning(self, "MiniPanel", "Select a valid root directory first.")
            return

        if not self._selected_files:
            self._refresh_discovered_files(auto_select=True)
        if not self._selected_files:
            QMessageBox.warning(self, "MiniPanel", "No PerFishMetrics workbooks are available to analyze.")
            return
        self._apply_reference_defaults()

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
                self.pbar.setValue(int((cur / total) * 100))
            self.pbar.setFormat(msg)

    def _on_analysis_error(self, error):
        self.btn_run.setEnabled(True)
        self.pbar.setFormat("Error")
        self._current_worker = None
        try:
            QMessageBox.critical(self, "MiniPanel Error", str(error))
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
                    "Output Directory": result.get("output_dir"),
                    "Panel Directory": result.get("panel_dir"),
                    "Stats XLSX": result.get("stats_xlsx"),
                    "Groups": result.get("n_groups"),
                    "Parameters": result.get("n_params"),
                    "CTRL Index (1-based)": result.get("ctrl_index_1based"),
                    "Reference Index (1-based)": result.get("reference_index_1based"),
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
                name="MiniPanel Summary",
                area="bottom",
            )
