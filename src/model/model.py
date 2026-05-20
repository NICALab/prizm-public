import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models


# Define the DeepLabV3+ model
class DeepLabV3(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        """DeepLabV3 model with ResNet50 backbone for semantic segmentation."""
        super(DeepLabV3, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False)
        out_channels = self.model.backbone.conv1.out_channels
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels if isinstance(in_channels, int) else 1,
            out_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(
            in_channels, num_classes, kernel_size=(1, 1)
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)["out"]


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model with ResNet50 backbone for semantic segmentation."""

    def __init__(
        self,
        num_classes,
        decoder_atrous_rates,
        backbone="resnet50",
        encoder_depth=5,
        decoder_channels=256,
        encoder_output_stride=16,
        in_channels=1,
        encoder_weights=None,
    ):
        super(DeepLabV3Plus, self).__init__()
        decoder_atrous_rates = tuple(decoder_atrous_rates)
        self.model = smp.DeepLabV3Plus(
            in_channels=in_channels,
            classes=num_classes,
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            encoder_output_stride=encoder_output_stride,
            decoder_atrous_rates=decoder_atrous_rates,
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


# # Main function
# if __name__ == "__main__":
#     # Initialize the model
#     model = DeepLabV3Plus(num_classes=3)

#     # Load a sample image
#     image = torch.randn(2, 1, 304, 304)

#     # Perform a forward pass
#     output = model(image)
#     print(output.shape)
