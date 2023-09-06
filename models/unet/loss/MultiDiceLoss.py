import torch
import torch.nn
from .DiceLoss import DiceLoss
import torch.nn.functional as F


# 获取类型标签的独热编码
# input shape [N, d1, ..., dn]
# output shape [N, C, d1, ..., dn]
class OneHot:
    def __init__(self, num_classes=-1) -> None:
        self.num_classes = num_classes

    def __call__(self, class_index_label: torch.Tensor) -> torch.Tensor:
        shape_len = len(class_index_label.size())
        class_one_hot_label: torch.Tensor = F.one_hot(
            class_index_label, self.num_classes
        )
        class_one_hot_label = class_one_hot_label.permute(
            0, shape_len, *range(1, shape_len)
        )
        return class_one_hot_label


class MultiDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, power=2, num_classes=-1):
        super().__init__()
        self.binary_dice = DiceLoss(smooth=smooth, power=power)
        self.softmax = torch.nn.Softmax(dim=1)
        self.get_one_hot = OneHot(num_classes=num_classes)

    def forward(self, result: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        result = self.softmax(result)
        label = self.get_one_hot(label)

        num_classes = label.size(1)
        total_dice_loss = 0
        for i in range(num_classes):
            total_dice_loss += self.binary_dice(result[:, i], label[:, i])

        return total_dice_loss / num_classes
