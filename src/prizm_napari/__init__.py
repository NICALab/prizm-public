__version__ = "0.0.1"

# Napari discovers these exports through the plugin manifest. Keeping the
# imports optional also allows non-GUI command-line modules to load without Qt.
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
    "PRIZMBatchSegmentationQWidget",
    "PRIZMMoAPredictionQWidget",
    "PRIZMMiniPanelQWidget",
)
