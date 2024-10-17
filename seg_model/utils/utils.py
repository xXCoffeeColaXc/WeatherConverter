from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):

    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_run():
    checkpoint_path = 'seg_model/outputs/checkpoints'
    sample_path = 'seg_model/outputs/samples'
    max_run_id = _find_max_run_id(checkpoint_path)

    new_run_id = max_run_id + 1
    run_dir = f'run_{new_run_id}'
    new_sample_path = os.path.join(sample_path, run_dir)
    new_checkpoint_path = os.path.join(checkpoint_path, run_dir)

    # Create new directories for checkpoints
    os.makedirs(new_sample_path, exist_ok=True)
    os.makedirs(new_checkpoint_path, exist_ok=True)

    print(f"New run directory created: {run_dir}")

    return run_dir


def _find_max_run_id(checkpoint_path):
    max_run_id = 0
    if os.path.exists(checkpoint_path):
        for dir_name in os.listdir(checkpoint_path):
            if dir_name.startswith('run_'):
                try:
                    run_id = int(dir_name.split('_')[1])
                    max_run_id = max(max_run_id, run_id)
                except ValueError:
                    pass  # Ignore directories with non-integer run numbers

    return max_run_id


def print_class_iou(epoch, epochs, val_score):
    print(f"[Val] Class IoU at Epoch [{epoch}/{epochs}] - Score:")

    # Loop through the dictionary and print each class with indentation
    for class_id, iou_value in val_score['Class IoU'].items():
        print(f"    Class {class_id}: {iou_value}")
