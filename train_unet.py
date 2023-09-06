import os
import json
import time
import logging
import random
import argparse
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms, InterpolationMode
import torchvision.transforms.functional as ttf
import models.unet.utils as utils
from models.unet.UNet import UNet
from models.unet.SE_UNet import SEUNet
from models.unet.loss.MultiDiceLoss import MultiDiceLoss


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, "image_names.json"), "r") as f:
            self.image_names = json.load(f)

    def __getitem__(self, index) -> tuple[Image.Image, Image.Image]:
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.dataset_dir, "xrays", image_name)).convert("L")
        mask = Image.open(os.path.join(self.dataset_dir, "masks", image_name))

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)


class Preload(Dataset):
    """
    wrap a dataset to preload all items eagerly
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.data = []
        for i in range(len(dataset)):
            self.data.append(dataset[i])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.dataset)


class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        flip: float = None,
        crop: float = None,
        rotate: list = None,
    ):
        self.dataset = dataset
        self.flip = flip
        self.crop = crop
        self.rotate = rotate

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        image, mask = TransformedDataset.data_transform(image, mask, self.flip, self.crop, self.rotate)
        return image, mask

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def data_transform(
        image: Image.Image, mask: Image.Image = None, flip: float = None, crop: float = None, rotate: list = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        convert PIL Image to torch Tensor and do some augmentation
        @param image: PIL Image
        @param mask: PIL Image
        @param flip: float, 0.0 ~ 1.0, probability of flip
        @param crop: float, 0.0 ~ 1.0, probability of crop
        @param rotate: list, [min_angle, max_angle], in degree
        """
        dummy_mask = mask if mask is not None else Image.new("L", image.size)
        # resize
        image = image.resize((256, 256), Image.BILINEAR)
        dummy_mask = dummy_mask.resize((256, 256), Image.NEAREST)

        # to tensor
        image = ttf.to_tensor(image)  # shape(1, 256, 256)
        dummy_mask = torch.from_numpy(np.array(dummy_mask)).long().unsqueeze(0)  # shape(1, 256, 256)

        # normalize
        image = ttf.normalize(image, [0.458], [0.173])

        # flip
        if flip is not None and random.random() < flip:
            image = ttf.hflip(image)
            dummy_mask = ttf.hflip(dummy_mask)

        # crop
        if crop is not None and random.random() < crop:
            size = random.randint(128, 225)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(size, size))
            image = ttf.crop(image, i, j, h, w)
            dummy_mask = ttf.crop(dummy_mask, i, j, h, w)

            # resize
            image = ttf.resize(image, (256, 256), InterpolationMode.BILINEAR)
            dummy_mask = ttf.resize(dummy_mask, (256, 256), InterpolationMode.NEAREST)

        # rotate
        if rotate is not None and random.random() < 0.1:
            angle = random.randint(rotate[0], rotate[1])
            image = ttf.rotate(image, angle)
            dummy_mask = ttf.rotate(dummy_mask, angle)

        dummy_mask = dummy_mask.squeeze(0)
        return image, dummy_mask


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # copy this file to output dir
    shutil.copy(__file__, output_dir)

    set_seeds(args.seed)
    cuda = args.cuda
    is_parallel = args.is_parallel

    num_classes = args.num_classes
    if args.model == "unet":
        model = UNet(in_channels=1, out_channels=num_classes + 1)
    else:
        model = SEUNet(n_cls=num_classes + 1)
    if cuda:
        model = model.cuda()
    if is_parallel:
        model = nn.DataParallel(model)

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.info("Loading dataset...")

    dataset_dir = args.dataset_dir
    dataset = Preload(SegmentationDataset(dataset_dir))
    logger.info("Loaded dataset!")
    dataset = TransformedDataset(dataset, flip=0.1, crop=0.1, rotate=[-10, 10])
    train_size = int(len(dataset) * args.train_ratio)
    validation_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(args.seed)
    )

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=1e-3)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    dice_loss_func = MultiDiceLoss()
    ce_loss_func = CrossEntropyLoss()

    tensorboard_writer = SummaryWriter(output_dir)

    if args.resume is not None:
        logger.warning("Resume not implemented yet, ignoring it!")

    min_valid_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info(f"Epoch: {epoch}")

        # train
        model.train()
        train_loss_dice = 0.0
        train_loss_ce = 0.0
        for i, (image, mask) in enumerate(train_loader):
            logger.info(f"Train batch: {i}/{len(train_loader)}")
            if cuda:
                image = image.cuda()
                mask = mask.cuda()

            optimizer.zero_grad()
            pred = model(image)
            loss_dice = dice_loss_func(pred, mask)
            loss_ce = ce_loss_func(pred, mask)
            loss = loss_dice + loss_ce
            loss.backward()
            optimizer.step()

            train_loss_dice += loss_dice.item()
            train_loss_ce += loss_ce.item()

        train_loss_dice /= len(train_loader)
        train_loss_ce /= len(train_loader)
        logger.info(f"Train loss: {train_loss_dice}, {train_loss_ce}")
        tensorboard_writer.add_scalar("train_loss/total", train_loss_dice + train_loss_ce, epoch)
        tensorboard_writer.add_scalar("train_loss/dice", train_loss_dice, epoch)
        tensorboard_writer.add_scalar("train_loss/ce", train_loss_ce, epoch)

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss_dice = 0.0
            valid_loss_ce = 0.0
            for i, (image, mask) in enumerate(val_loader):
                logger.info(f"Validation batch: {i}/{len(val_loader)}")
                if cuda:
                    image = image.cuda()
                    mask = mask.cuda()

                pred = model(image)
                loss_dice = dice_loss_func(pred, mask)
                loss_ce = ce_loss_func(pred, mask)

                valid_loss_dice += loss_dice.item()
                valid_loss_ce += loss_ce.item()

            valid_loss_dice /= len(val_loader)
            valid_loss_ce /= len(val_loader)
            logger.info(f"Validation loss: {valid_loss_dice}, {valid_loss_ce}")
            tensorboard_writer.add_scalar("validation_loss/total", valid_loss_dice + valid_loss_ce, epoch)
            tensorboard_writer.add_scalar("validation_loss/dice", valid_loss_dice, epoch)
            tensorboard_writer.add_scalar("validation_loss/ce", valid_loss_ce, epoch)

            if valid_loss_dice + valid_loss_ce < min_valid_loss:
                min_valid_loss = valid_loss_dice + valid_loss_ce
                utils.save_state(
                    model=model,
                    out_dir=output_dir,
                    checkpoint_name=f"epoch_{epoch}_loss_{min_valid_loss}.pth",
                    batch_size=batch_size,
                    epoch=epoch,
                    is_parallel=is_parallel,
                )
                logger.info(f"Save model: epoch_{epoch}_loss_{min_valid_loss}.pth")

        # scheduler.step()

        if epoch % 5 == args.save_interval:
            utils.save_state(
                model=model,
                out_dir=output_dir,
                checkpoint_name=f"auto_save.pth",
                batch_size=batch_size,
                epoch=epoch,
                is_parallel=is_parallel,
            )

    tensorboard_writer.close()
    utils.save_state(
        model=model,
        out_dir=output_dir,
        checkpoint_name=f"last_epoch.pth",
        batch_size=batch_size,
        epoch=epoch,
        is_parallel=is_parallel,
    )
    logger.info(f"Save model: last_epoch.pth")
    logger.info("Done!")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--is_parallel", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--model", type=str, choices=["unet", "seunet"], default="unet")
    parser.add_argument("--num_classes", type=int, help="number of classes, not including background")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--save_interval", type=int, default=5)
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
