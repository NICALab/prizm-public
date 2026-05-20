__version__ = "0.0.1"

# Lazy imports to avoid loading napari dependencies when CLI is used
# These will only be imported when the package is used in napari context
def _lazy_import_widgets():
    """Lazy import widgets only when needed (e.g., in napari)."""
    from ._widget import PRIZMBatchSegmentationQWidget
    from .moa_widget import PRIZMMoAPredictionQWidget
    from .minipanel_widget import PRIZMMiniPanelQWidget
    return (
        PRIZMBatchSegmentationQWidget,
        PRIZMMoAPredictionQWidget,
        PRIZMMiniPanelQWidget,
    )

# For backward compatibility and napari plugin discovery, we still need to import
# But we'll do it in a try-except to allow CLI usage without napari
try:
    from ._widget import PRIZMBatchSegmentationQWidget
    from .moa_widget import PRIZMMoAPredictionQWidget
    from .minipanel_widget import PRIZMMiniPanelQWidget
except ImportError:
    # If napari is not available, these will be None
    # This allows CLI to work without napari installed
    PRIZMBatchSegmentationQWidget = None
    PRIZMMoAPredictionQWidget = None
    PRIZMMiniPanelQWidget = None

__all__ = (
    # "PRIZMSegmentationQWidget",
    "PRIZMBatchSegmentationQWidget",
    "PRIZMMoAPredictionQWidget",
    "PRIZMMiniPanelQWidget",
)
