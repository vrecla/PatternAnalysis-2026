"""Model components for ADNI AD-vs-NC classification.

Contains three models used in the benchmark progression MLP -> CNN -> ConvNeXt:

* ``MLPBaseline``  - fully connected network on a downsampled image.
* ``SimpleCNN``    - a plain VGG-style convolutional baseline.
* ``ConvNeXt``     - a reduced-size ConvNeXt (Liu et al., CVPR 2022, "A ConvNet
                     for the 2020s") written from scratch, trained from scratch.

Only ``torch`` / ``torch.nn`` are used here (no NumPy), as the assignment
requires. The ConvNeXt blocks are implemented by hand rather than imported from
torchvision.

ConvNeXt design elements retained from the paper:
  1. "Patchify" stem: a strided convolution instead of a conv + max-pool stem.
  2. Depthwise 7x7 convolution as the spatial mixer (large-kernel, like
     attention's wide receptive field but with a convolution's inductive bias).
  3. Inverted-bottleneck MLP (1x1 expand x4 -> GELU -> 1x1 project).
  4. LayerNorm (not BatchNorm), one GELU, one norm per block.
  5. Layer scale and stochastic depth (DropPath) for stable training.
  6. Separate downsampling layers (LayerNorm + 2x2 stride-2 conv) between stages.

Reduction versus the paper's ConvNeXt-T (96/192/384/768 channels, depths 3-3-9-3):
fewer channels and blocks so it trains from scratch on a modest medical dataset
within Rangpur time limits. The architecture, not the size, is what is kept.
"""

from typing import Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# ConvNeXt building blocks
# --------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor.

    nn.LayerNorm normalises the last dimension, so for channels-first image
    tensors we normalise across C at each spatial position manually.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class DropPath(nn.Module):
    """Stochastic depth: randomly skip a residual branch per sample (training only).

    Surviving samples are rescaled by 1/keep_prob so the expected value of the
    branch output is unchanged, which keeps train and eval activations consistent.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over all other dimensions.
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob).to(x.dtype)
        return x * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    """One ConvNeXt block: dw7x7 -> LN -> 1x1 expand -> GELU -> 1x1 project -> scale.

    The two 1x1 convolutions play the role of the paper's pointwise (Linear)
    layers; using 1x1 convs keeps the tensor in channels-first layout so no
    permutes are needed.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        expansion: int = 4,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw_expand = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw_project = nn.Conv2d(expansion * dim, dim, kernel_size=1)
        # Layer scale: a learnable per-channel multiplier initialised near zero so
        # each block starts as an almost-identity mapping, easing optimisation.
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        x = x * self.gamma[None, :, None, None]
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    """Reduced-size ConvNeXt for single-channel 2D medical slices.

    Args:
        in_channels: input channels (1 for greyscale MRI slices).
        num_classes: number of output logits (2 for AD vs NC).
        dims: channel width of each stage.
        depths: number of ConvNeXtBlocks in each stage.
        drop_path_rate: maximum stochastic-depth rate, increased linearly with depth.
        layer_scale_init: initial value of the layer-scale parameter.
        head_dropout: dropout before the classifier (helps calibration on small data).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dims: Sequence[int] = (48, 96, 192, 384),
        depths: Sequence[int] = (2, 2, 6, 2),
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError("dims and depths must have the same length")

        # Patchify stem: non-overlapping 4x4 patches via a stride-4 convolution.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Stochastic depth rate grows linearly from 0 to drop_path_rate over all blocks.
        total_blocks = sum(depths)
        rates = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

        stages = []
        block_idx = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            layers = []
            if i > 0:
                # Separate downsampling layer between stages (norm then 2x2 stride-2 conv).
                layers += [LayerNorm2d(dims[i - 1]), nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2)]
            for _ in range(depth):
                layers.append(ConvNeXtBlock(dim, rates[block_idx], layer_scale_init))
                block_idx += 1
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled, normalised feature vector (N, dims[-1])."""
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean(dim=(-2, -1))  # global average pooling
        return self.head_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_dropout(self.forward_features(x)))


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """Fully connected baseline on a downsampled, flattened image.

    The image is average-pooled to ``pool_size`` x ``pool_size`` first so the
    first linear layer stays a reasonable size. It has no spatial inductive
    bias, which makes it the weakest reference point in the progression.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        pool_size: int = 32,
        hidden: Sequence[int] = (512, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        layers = []
        prev = in_channels * pool_size * pool_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(self.pool(x), 1))


class SimpleCNN(nn.Module):
    """Plain VGG-style CNN baseline: [conv-BN-ReLU] x2 + max-pool, four times.

    Deliberately conventional (BatchNorm, ReLU, max-pooling, 3x3 convs) so the
    comparison with ConvNeXt isolates the effect of the modernised design.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        widths: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = in_channels
        for w in widths:
            layers += [
                nn.Conv2d(prev, w, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = w
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters, for the resource-profiling table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Construct a model by name: 'mlp', 'cnn' or 'convnext'."""
    name = name.lower()
    if name == "mlp":
        return MLPBaseline(num_classes=num_classes, **kwargs)
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    if name == "convnext":
        return ConvNeXt(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose from: mlp, cnn, convnext")


if __name__ == "__main__":
    # Quick self-check: shapes and parameter counts on a dummy batch.
    dummy = torch.randn(2, 1, 224, 224)
    for model_name in ("mlp", "cnn", "convnext"):
        net = build_model(model_name)
        print(f"{model_name:9s} out={tuple(net(dummy).shape)} params={count_parameters(net):,}")
