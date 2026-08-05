import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(
            num_features=output_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = self.batch_norm(outputs)
        outputs = self.relu(outputs)

        return outputs


class TeacherCNN(nn.Module):
    def __init__(
        self,
        number_of_classes: int = 10,
    ) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
        )

        self.global_average_pooling = nn.AdaptiveAvgPool2d(
            output_size=1,
        )

        self.classifier = nn.Linear(
            in_features=256,
            out_features=number_of_classes,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.stage1(inputs)
        features = self.stage2(features)
        features = self.stage3(features)

        pooled_features = self.global_average_pooling(
            features
        )

        flattened_features = torch.flatten(
            pooled_features,
            start_dim=1,
        )

        logits = self.classifier(
            flattened_features
        )

        return logits


class StudentCNN(nn.Module):
    def __init__(
        self,
        number_of_classes: int = 10,
    ) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 32),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128),
        )

        self.global_average_pooling = nn.AdaptiveAvgPool2d(
            output_size=1,
        )

        self.classifier = nn.Linear(
            in_features=128,
            out_features=number_of_classes,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.stage1(inputs)
        features = self.stage2(features)
        features = self.stage3(features)

        pooled_features = self.global_average_pooling(
            features
        )

        flattened_features = torch.flatten(
            pooled_features,
            start_dim=1,
        )

        logits = self.classifier(
            flattened_features
        )

        return logits


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
