import numpy as np
import matplotlib.pyplot as plt
import logging, os
import random
import torch

from scipy import signal
def resize(arr, new_len):
    if len(arr.shape) > 1:
        arr = np.squeeze(arr)
    length = len(arr)
    indices_new = np.linspace(0, length-1, new_len)
    indices_old = np.arange(0, length)
    return np.interp(indices_new, indices_old, arr)


def find_diff_min(data):
    kernel_len = 5
    data = signal.convolve(data, np.ones(kernel_len) / kernel_len)
    Sregion = len(data) // 4
    first_diff = - np.diff(data)
    min_diff_idx = np.argmax(first_diff[:Sregion])

    return min_diff_idx + 1

def load_train_test_dict(hp_dict, ratio_train=0.7):
    ## split the hp_dict into train and test
    seed_everything()

    items = list(hp_dict.items())
    random.shuffle(items)

    split_index = int(len(items) * ratio_train)

    train = dict(items[:split_index])
    test = dict(items[split_index:])

    return train, test

def count_number_parameters( model ):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters

def plot_x(x, save_path=None, axis_off=False):
    figure, axis = plt.subplots(figsize=(10, 10), dpi=30)
    axis.set_ylim([0, 0.8])
    axis.set_xlim([0, len(x)])
    if axis_off:
        axis.set_axis_off()
    axis.plot(x, linewidth=1.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()


def plot_stacked(sigs, save_dir):
    plt.figure(figsize=(10, 8))
    plt.ylim((0, 0.8))
    plt.xlim((0, len(sigs[0])))
    sigs = sigs.copy()
    np.random.shuffle(sigs)
    sigs_to_print = sigs[:3000]
    if len(sigs_to_print) > 3000:
        alpha = 0.01
    else:
        alpha = 1 - 0.99 * (len(sigs_to_print) / 3000) ** 0.01
    for i, sig in enumerate(sigs_to_print):
        plt.plot(sig, color='black', linewidth=1, alpha=alpha*0.7)
    plt.savefig(f'{save_dir}.png')
    plt.close()


def get_logger(exp_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    sh = logging.StreamHandler()
    fa = logging.Formatter("%(asctime)s %(message)s")
    fh.setFormatter(fa)
    sh.setFormatter(fa)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


DEFAULT_RANDOM_SEED = 2023


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)