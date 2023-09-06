import os
import torch
import torch.nn
from .UNet import UNet
from .SE_UNet import SEUNet


def load_state(
    *,
    checkpoint_path: str,
    cuda: bool = True,
):
    device = torch.device("cuda" if cuda else "cpu")
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint_obj["model_state_dict"]
    epoch = checkpoint_obj["epoch"]
    batch_size = checkpoint_obj["batch_size"]

    return model_state_dict, epoch, batch_size


def save_state(
    *,
    model: torch.nn.Module,
    out_dir: str,
    checkpoint_name: str,
    batch_size: int,
    epoch: int,
    is_parallel: bool = False,
):
    checkpoint_path = os.path.join(out_dir, checkpoint_name)
    model_state_dict = model.module.state_dict() if is_parallel else model.state_dict()
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "epoch": epoch,
            "batch_size": batch_size,
        },
        checkpoint_path,
    )


def load_unet(checkpoint_path, out_channels, cuda=True):
    model = UNet(in_channels=1, out_channels=out_channels)
    if cuda:
        model.cuda()
    model_state, _, _ = load_state(checkpoint_path=checkpoint_path, cuda=cuda)
    model.load_state_dict(model_state)
    return model


def load_seunet(checkpoint_path, out_channels, cuda=True):
    model = SEUNet(n_cls=out_channels)
    if cuda:
        model.cuda()
    model_state, _, _ = load_state(checkpoint_path=checkpoint_path, cuda=cuda)
    model.load_state_dict(model_state)
    return model
