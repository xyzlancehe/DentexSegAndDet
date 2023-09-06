import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.double_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv_block(x)


class UNetDownwardLayer(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_block = UNetDoubleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.down_sample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out_forward = self.conv_block(x)
        out_downward = self.down_sample(out_forward)
        return out_forward, out_downward


class UNetUpwardLayer(torch.nn.Module):
    def __init__(self, *, num_features) -> None:
        super().__init__()
        self.conv_block = UNetDoubleConvBlock(
            in_channels=num_features * 2,
            out_channels=num_features,
        )
        self.up_sample_conv = nn.ConvTranspose2d(
            in_channels=num_features * 2,
            out_channels=num_features,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x_from_lower_layer, x_from_encoder_forward):
        x_upsampled = self.up_sample_conv(x_from_lower_layer)
        x = torch.cat(
            (x_upsampled, x_from_encoder_forward),
            dim=1,
        )
        return self.conv_block(x)


class UNet(torch.nn.Module):
    def __init__(
        self, *, in_size=(512, 512), in_channels=1, out_channels=1, init_features=64
    ) -> None:
        super().__init__()
        self.encode_down_layer1 = UNetDownwardLayer(
            in_channels=in_channels,
            out_channels=init_features,
        )
        self.encode_down_layer2 = UNetDownwardLayer(
            in_channels=init_features,
            out_channels=init_features * 2,
        )
        self.encode_down_layer3 = UNetDownwardLayer(
            in_channels=init_features * 2,
            out_channels=init_features * 4,
        )
        self.encode_down_layer4 = UNetDownwardLayer(
            in_channels=init_features * 4,
            out_channels=init_features * 8,
        )
        self.bottom_layer = UNetDoubleConvBlock(
            in_channels=init_features * 8,
            out_channels=init_features * 16,
        )
        self.decode_up_layer4 = UNetUpwardLayer(num_features=init_features * 8)
        self.decode_up_layer3 = UNetUpwardLayer(num_features=init_features * 4)
        self.decode_up_layer2 = UNetUpwardLayer(num_features=init_features * 2)
        self.decode_up_layer1 = UNetUpwardLayer(num_features=init_features)
        self.output_conv = nn.Conv2d(
            in_channels=init_features,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.out_channels = out_channels
        if out_channels == 1:
            self.out_layer_func = nn.Sigmoid()
        else:
            self.out_layer_func = nn.Identity()

    def forward(self, x):
        x_encode_forward1, x_encode_downward1 = self.encode_down_layer1(x)

        x_encode_forward2, x_encode_downward2 = self.encode_down_layer2(
            x_encode_downward1
        )

        x_encode_forward3, x_encode_downward3 = self.encode_down_layer3(
            x_encode_downward2
        )

        x_encode_forward4, x_encode_downward4 = self.encode_down_layer4(
            x_encode_downward3
        )

        x_out_bottom = self.bottom_layer(x_encode_downward4)

        x_decode_upward4 = self.decode_up_layer4(x_out_bottom, x_encode_forward4)

        x_decode_upward3 = self.decode_up_layer3(x_decode_upward4, x_encode_forward3)

        x_decode_upward2 = self.decode_up_layer2(x_decode_upward3, x_encode_forward2)

        x_out_decode = self.decode_up_layer1(x_decode_upward2, x_encode_forward1)

        x_out_result = self.out_layer_func(self.output_conv(x_out_decode))

        return x_out_result
