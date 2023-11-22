import torch
from torch import nn

def simple_mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    zero_init=False,
    norm=True,
):
    """MLP layers
    args:
        input_size (int): dim of inputs
        layer_sizes (list): dim of hidden layers
        output_size (int): dim of outputs
        init_zero (bool): zero initialization for the last layer (including w and b).
            This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1], momentum=momentum))
            layers.append(act())
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    if zero_init:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        zero_init=False,
        norm=False,
        skip_connection=False,
    ):
        super().__init__()

        self.skip_connection = skip_connection

        sizes = [input_size] + layer_sizes + [output_size]
        self.layer_n = len(layer_sizes) + 1
        self.layers = nn.ModuleList()
        self.act = activation()
        self.output_act = output_activation()
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            if self.skip_connection and i >= 1:
                in_size += input_size
            layer = [nn.Linear(in_size, out_size)]
            if norm:
                layer.append(nn.BatchNorm1d(out_size, momentum=momentum))
            self.layers.append(nn.Sequential(*layer))

        if zero_init:
            self.layers[-1][0].weight.data.fill_(0)
            self.layers[-1][0].bias.data.fill_(0)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < self.layer_n - 1:
                out = self.act(out)
            else:
                out = self.output_act(out)
            if self.skip_connection and i < self.layer_n - 1:
                out = torch.cat((out, x), dim=-1)
        return out
        
def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes, outplanes=None, stride=1, downsample=None, disable_bn=False
    ):
        super().__init__()
        if outplanes is None:
            outplanes = inplanes
        if disable_bn:
            norm_layer = nn.Identity
        else:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.skip_conv = outplanes != inplanes
        self.stride = stride
        if outplanes != inplanes:
            if downsample is None:
                self.conv3 = conv1x1(inplanes, outplanes)
            else:
                self.conv3 = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv:
            out += self.conv3(identity)
        else:
            out += identity
        out = self.relu(out)
        return out