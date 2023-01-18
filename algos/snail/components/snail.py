import torch
import torch.nn as nn


class CasualConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()

        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()


class TCBlock(nn.Module):
    def __init__(self):
        super().__init__()


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
