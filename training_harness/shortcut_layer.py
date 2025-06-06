import torch.nn as nn

class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Implemented as linear layer for now.
        """
        super().__init__()
        self.out_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.out_proj(x)
        