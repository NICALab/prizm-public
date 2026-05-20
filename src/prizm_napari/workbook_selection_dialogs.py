from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from prizm_napari.input_discovery import infer_perfish_group_name


def workbook_display_label(path: str | Path, root: str | Path | None = None) -> str:
    fp = Path(path)
    if root is not None:
        try:
            rel = fp.relative_to(Path(root))
        except Exception:
            rel = fp
    else:
        rel = fp
    group = infer_perfish_group_name(fp)
    rel_text = rel.as_posix()
    return f"{group} | {rel_text}"


class OrderedWorkbookSelectionDialog(QDialog):
    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        title: str,
        prompt: str,
        root: str | Path | None = None,
        initial_checked: Optional[Sequence[str | Path]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 520)

        initial = {str(Path(p).resolve()) for p in (initial_checked or [])}
        self._root = Path(root) if root is not None else None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt, self))

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        for fp_raw in files:
            fp = Path(fp_raw).resolve()
            item = QListWidgetItem(workbook_display_label(fp, self._root))
            item.setData(Qt.UserRole, str(fp))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if str(fp) in initial else Qt.Unchecked)
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget)

        button_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All", self)
        self.btn_select_all.clicked.connect(self._select_all)
        button_row.addWidget(self.btn_select_all)

        self.btn_clear_all = QPushButton("Clear All", self)
        self.btn_clear_all.clicked.connect(self._clear_all)
        button_row.addWidget(self.btn_clear_all)

        self.btn_move_up = QPushButton("Move Up", self)
        self.btn_move_up.clicked.connect(lambda: self._move_current(-1))
        button_row.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton("Move Down", self)
        self.btn_move_down.clicked.connect(lambda: self._move_current(1))
        button_row.addWidget(self.btn_move_down)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _select_all(self):
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(Qt.Checked)

    def _clear_all(self):
        for idx in range(self.list_widget.count()):
            self.list_widget.item(idx).setCheckState(Qt.Unchecked)

    def _move_current(self, delta: int):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        new_row = row + int(delta)
        if new_row < 0 or new_row >= self.list_widget.count():
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new_row, item)
        self.list_widget.setCurrentRow(new_row)

    def selected_paths(self):
        selected = []
        for idx in range(self.list_widget.count()):
            item = self.list_widget.item(idx)
            if item.checkState() == Qt.Checked:
                selected.append(Path(item.data(Qt.UserRole)))
        return selected

    @staticmethod
    def get_selected(
        files: Sequence[str | Path],
        *,
        title: str,
        prompt: str,
        root: str | Path | None = None,
        initial_checked: Optional[Sequence[str | Path]] = None,
        parent=None,
    ):
        dlg = OrderedWorkbookSelectionDialog(
            files,
            title=title,
            prompt=prompt,
            root=root,
            initial_checked=initial_checked,
            parent=parent,
        )
        if dlg.exec_() == QDialog.Accepted:
            return dlg.selected_paths()
        return None


class GroupNameEditorDialog(QDialog):
    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        title: str,
        root: str | Path | None = None,
        initial_groups: Optional[Dict[str, str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 420)
        self._files = [Path(p).resolve() for p in files]
        self._root = Path(root) if root is not None else None
        self._initial_groups = {str(Path(k).resolve()): str(v) for k, v in (initial_groups or {}).items()}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Edit the training group names for the non-vehicle files.", self))

        self.table = QTableWidget(len(self._files), 2, self)
        self.table.setHorizontalHeaderLabels(["Workbook", "Group"])
        for row, fp in enumerate(self._files):
            label_item = QTableWidgetItem(workbook_display_label(fp, self._root))
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, label_item)

            default_group = self._initial_groups.get(str(fp), infer_perfish_group_name(fp))
            self.table.setItem(row, 1, QTableWidgetItem(default_group))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def group_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row, fp in enumerate(self._files):
            cell = self.table.item(row, 1)
            group = cell.text().strip() if cell is not None else ""
            if not group:
                group = infer_perfish_group_name(fp)
            out[str(fp)] = group
        return out

    @staticmethod
    def get_group_map(
        files: Sequence[str | Path],
        *,
        title: str,
        root: str | Path | None = None,
        initial_groups: Optional[Dict[str, str]] = None,
        parent=None,
    ):
        dlg = GroupNameEditorDialog(
            files,
            title=title,
            root=root,
            initial_groups=initial_groups,
            parent=parent,
        )
        if dlg.exec_() == QDialog.Accepted:
            return dlg.group_map()
        return None


class MiniPanelReferenceDialog(QDialog):
    def __init__(
        self,
        group_names: Sequence[str],
        *,
        default_control: str,
        default_reference: str,
        include_reference_in_heatmap: bool,
        save_all_pairs_excel: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pick MiniPanel Control / Reference")
        self.resize(560, 280)

        groups = [str(g) for g in group_names]
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Choose the control group and the reference group used for MiniPanel statistics and heatmap comparisons.",
                self,
            )
        )

        self.cb_control = QComboBox(self)
        self.cb_control.setEditable(False)
        self.cb_control.addItems(groups)
        if default_control in groups:
            self.cb_control.setCurrentText(default_control)
        layout.addWidget(QLabel("Control Group", self))
        layout.addWidget(self.cb_control)

        self.cb_reference = QComboBox(self)
        self.cb_reference.setEditable(False)
        self.cb_reference.addItems(groups)
        if default_reference in groups:
            self.cb_reference.setCurrentText(default_reference)
        layout.addWidget(QLabel("Reference Group", self))
        layout.addWidget(self.cb_reference)

        self.chk_include_reference = QCheckBox("Include reference group in heatmap", self)
        self.chk_include_reference.setChecked(bool(include_reference_in_heatmap))
        layout.addWidget(self.chk_include_reference)

        self.chk_save_all_pairs = QCheckBox("Save all pairwise Welch t-tests", self)
        self.chk_save_all_pairs.setChecked(bool(save_all_pairs_excel))
        layout.addWidget(self.chk_save_all_pairs)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def selection(self):
        return {
            "control_group_name": self.cb_control.currentText().strip(),
            "reference_group_name": self.cb_reference.currentText().strip(),
            "include_reference_in_heatmap": self.chk_include_reference.isChecked(),
            "save_all_pairs_excel": self.chk_save_all_pairs.isChecked(),
        }

    @staticmethod
    def get_selection(
        group_names: Sequence[str],
        *,
        default_control: str,
        default_reference: str,
        include_reference_in_heatmap: bool,
        save_all_pairs_excel: bool,
        parent=None,
    ):
        dlg = MiniPanelReferenceDialog(
            group_names,
            default_control=default_control,
            default_reference=default_reference,
            include_reference_in_heatmap=include_reference_in_heatmap,
            save_all_pairs_excel=save_all_pairs_excel,
            parent=parent,
        )
        if dlg.exec_() == QDialog.Accepted:
            return dlg.selection()
        return None
