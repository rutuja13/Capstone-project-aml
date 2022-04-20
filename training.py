import os
import cv2
import random
import numpy as np
from collections import deque
import albumentations as A
from skimage.transform import rotate

import torch
import torch.nn.functional as F

from metrics import Dice


class Am:
    """Simple average meter which stores progress as a running average"""

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self._val, self._sum, self._count, self._running = [0] * 4
        self.avg, self.running_avg = 0, -1
        self.reset()

    def reset(self):
        self._val = 0
        self._sum = 0
        self._count = 0
        self._running = deque(maxlen=self.n_for_running_average)
        self.avg = 0
        self.running_avg = -1

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.detach().cpu().numpy()
        self._val = val
        self._sum += val * n
        self._running.extend([val] * n)
        self._count += n
        self.avg = self._sum / self._count
        self.running_avg = sum(self._running) / len(self._running)


def cycle(train_val_test, model, dataloader, device, epoch, criterion, optimizer, scheduler=None):
    # Create some meters to measure our progress
    meter_loss, meter_acc, meter_precision, meter_recall, meter_f1, meter_support = Am(), Am(), Am(), Am(), Am(), Am()

    if train_val_test == 'train':
        model.train()
        training = True
    elif train_val_test in ('val', 'test'):
        model.eval()
        training = False
    else:
        raise ValueError(f"train_val_test must be 'train', 'val', or 'test', not {train_val_test}")

    for i_batch, (x, y_true, _filepath, _label_name) in enumerate(dataloader):
        optimizer.zero_grad()

        # move samples to GPU is available
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # forward pass
        if training:
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y_true)

        # backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        # metrics
        with torch.no_grad():
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct_class = y_true == y_pred_class
            acc = torch.mean(correct_class * 1.0)

        meter_loss.update(loss)
        meter_acc.update(acc)

    return meter_loss.avg, meter_acc.avg


def load_classifier_transforms():
    print("M")
    transforms_train = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.33, 1.0), ratio=(0.95, 1.05), p=1.0),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast()
    ])
    print("A")
    transforms_test = A.Compose([
        A.LongestMaxSize(max_size=384),
        A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
    ])
    print("P")
    return transforms_train, transforms_test


def save_state(state, save_path, test_metric, best_metric, last_save_path, lowest_best=True, force=False):
    if force:
        torch.save(state, save_path)
    if (test_metric < best_metric) == lowest_best:
        if last_save_path:
            try:
                os.remove(last_save_path)
            except FileNotFoundError:
                print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    return best_metric, last_save_path


def cycle_seg(train_val_test, model, dataloader, device, epoch, criterion, optimizer, scheduler=None, log_freq=30):
    # Create some meters to measure our progress
    meter_loss, meter_acc = Am(), Am()
    meter_dice = Dice()

    if train_val_test == 'train':
        model.train()
        training = True
    elif train_val_test in ('val', 'test'):
        model.eval()
        training = False
    else:
        raise ValueError(f"train_val_test must be 'train', 'val', or 'test', not {train_val_test}")

    for i_batch, (x, y_true, _filepath, _label_name) in enumerate(dataloader):
        optimizer.zero_grad()

        # move samples to GPU is available
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # forward pass
        if training:
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y_true)

        # backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        # metrics
        with torch.no_grad():
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct_class = y_true == y_pred_class
            acc = torch.mean(correct_class * 1.0)

        meter_loss.update(loss)
        meter_acc.update(acc)
        meter_dice.accumulate(y_pred, y_true)

        # Loss intra-epoch printing
        if (i_batch + 1) % log_freq == 0:
            print(f"{train_val_test.upper(): >5} [{i_batch + 1:04d}/{len(dataloader):04d}]"
                  f"\t\tLOSS: {meter_loss.running_avg:.4f}"
                  f"\t\tDICE: {meter_dice.value:.4f}")

    return meter_loss.avg, meter_acc.avg, meter_dice


def contrast_gradient(image, **kwargs):
    height, width = image.shape[-2:]
    gradient = np.ones_like(image, dtype=np.float32)
    random_grad = random.choice([1.2, 1.3, 1.5, 2, 4, 5, 10, 20, 500, 500, 500, 500, 500])
    for i in range(height * 2):
        gradient[i:] -= 1 / (height * random_grad)
    random_angle = random.randint(0, 360)
    gradient = rotate(gradient, random_angle, resize=False, mode='edge')
    image = (image * gradient).astype('float32')
    return image


def load_segmentation_transforms():
    transforms_train = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.33, 1.0), ratio=(0.95, 1.05), p=1.0),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(),
        A.Lambda(image=contrast_gradient, p=0.5)
    ])

    transforms_test = A.Compose([
        A.LongestMaxSize(max_size=384),
        A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
    ])

    return transforms_train, transforms_test


def cycle_seg_classifier(train_val_test, model, unet_model, dataloader, device, criterion, optimizer, scheduler=None,
                         stack=True, scale_up_mask=False, clahe_transform=None):
    # Create some meters to measure our progress
    meter_loss, meter_acc, meter_precision, meter_recall, meter_f1, meter_support = Am(), Am(), Am(), Am(), Am(), Am()

    if train_val_test == 'train':
        model.train()
        training = True
    elif train_val_test in ('val', 'test'):
        model.eval()
        training = False
    else:
        raise ValueError(f"train_val_test must be 'train', 'val', or 'test', not {train_val_test}")

    for i_batch, (x, y_true, _filepath, _label_name) in enumerate(dataloader):
        optimizer.zero_grad()

        # move samples to GPU if available
        x = x.to(device, non_blocking=True)
        mask = torch.sigmoid(unet_model(x)) > 0.5
        y_true = y_true.to(device, non_blocking=True)

        with torch.no_grad():

            if stack:
                x_m = x.clone()
                x_m[~mask] = 0
                x = torch.cat((x_m, x), dim=1)  # Mask first as the scaling code will use this channel
            else:
                x[~mask] = 0

            if scale_up_mask:
                for i_img, img in enumerate(x):
                    col_from = (img[0] != 0).any(dim=1).nonzero()[0]
                    col_to = (img[0] != 0).any(dim=1).nonzero()[-1]
                    row_from = (img[0] != 0).any(dim=1).nonzero()[0]
                    row_to = (img[0] != 0).any(dim=1).nonzero()[-1]
                    crop = img[:, row_from:row_to, col_from:col_to]
                    x[i_img] = F.interpolate(crop.unsqueeze(0), img.size()[-2:], mode='bilinear')[0]

            if clahe_transform is not None:
                for i_img, img in enumerate(x):
                    x[i_img][0] = torch.tensor(clahe_transform(image=(x[i_img][0].cpu().numpy() * 255).astype(np.uint8))['image']/255).float()

        # forward pass
        if training:
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y_true)

        # backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        # metrics
        with torch.no_grad():
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct_class = y_true == y_pred_class
            acc = torch.mean(correct_class * 1.0)

        meter_loss.update(loss)
        meter_acc.update(acc)

    return meter_loss.avg, meter_acc.avg
