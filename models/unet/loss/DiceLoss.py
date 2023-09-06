import torch
import torch.nn


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, power=2):
        super().__init__()
        self.smooth = smooth
        self.power = power

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1) ** self.power
        y = y.view(batch_size, -1)

        intersection = (x * y).sum(1)
        unionset = x.sum(1) + y.sum(1)
        dice = (2 * intersection + self.smooth) / (unionset + self.smooth)
        dice = dice.sum() / batch_size
        loss = 1 - dice
        return loss
