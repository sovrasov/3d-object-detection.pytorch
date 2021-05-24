import pickle
import sys
import os
import os.path as osp
import errno
import random
import warnings
from collections import OrderedDict
from functools import partial
from importlib import import_module
from pprint import pformat

import numpy as np
import cv2 as cv
import torch

from addict import Dict as adict

from objectron.dataset import graphics
#pylint: disable = consider-using-with

OBJECTRON_CLASSES = ('bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_snap(model, optimizer, scheduler, epoch, log_path):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch': epoch}

    snap_name = f'{log_path}/snap_{epoch}.pth'
    print(f'==> saving checkpoint to {snap_name}')
    torch.save(checkpoint, snap_name)

def read_py_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    if not check_isfile(filename):
        raise RuntimeError("config not found")
    assert filename.endswith('.py')
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = adict({
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    })

    return cfg_dict

def load_checkpoint(fpath):
    r"""Loads checkpoint. Imported from openvinotoolkit/deep-object-reid.


    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers):
    if discarded_layers:
        print(
            '** The following layers are discarded '
            'due to unmatched keys or layer size: {}'.
            format(pformat(discarded_layers))
        )
    if unmatched_layers:
        print(
            '** The following layers were not loaded from checkpoint: {}'.
            format(pformat(unmatched_layers))
        )

def load_pretrained_weights(model, file_path='', pretrained_dict=None, extra_prefix=''):
    r"""Loads pretrianed weights to model. Imported from openvinotoolkit/deep-object-reid.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        file_path (str): path to pretrained weights.
    """
    def _remove_prefix(key, prefix):
        prefix = prefix + '.'
        if key.startswith(prefix):
            key = key[len(prefix):]
        return key

    if file_path:
        check_isfile(file_path)
    checkpoint = (load_checkpoint(file_path)
                       if not pretrained_dict
                       else pretrained_dict)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        k = extra_prefix + _remove_prefix(k, 'module')

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    message = file_path if file_path else "pretrained dict"
    unmatched_layers = sorted(set(model_dict.keys()) - set(new_state_dict))
    if len(matched_layers) == 0:
        print(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually'.format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

        raise RuntimeError(f'The pretrained weights {message} cannot be loaded')
    print(
        'Successfully loaded pretrained weights from "{}"'.
        format(message)
    )
    _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

def resume_from(model, chkpt_path, optimizer=None, scheduler=None):
    print(f'Loading checkpoint from "{chkpt_path}"')
    checkpoint = load_checkpoint(chkpt_path)
    if 'state_dict' in checkpoint:
        load_pretrained_weights(model, pretrained_dict=checkpoint['state_dict'])
    else:
        load_pretrained_weights(model, pretrained_dict=checkpoint)
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Loaded scheduler')
    if 'epoch' in checkpoint:
        # since saved in the end of an epoch we increment it by 1
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
        print("Warning. The epoch has not been restored.")
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch

def unnormalize_img(img,
                mean=[0.5931, 0.4690, 0.4229],
                std=[0.2471, 0.2214, 0.2157],
                tensor=True):
    if tensor:
        mean = torch.tensor(mean, device=img.device) * 255
        std = torch.tensor(std, device=img.device) * 255
        return img.permute(1, 2, 0)*std + mean
    return img*std + mean

def unnormalize(image_shape, normalized_keypoints):
    ''' transform image to global pixel values '''
    assert len(image_shape) in [2,3]
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape

    keypoints = np.multiply(normalized_keypoints, np.asarray([w, h], np.float32)).astype(int)
    return keypoints

def normalize(image_shape, unnormalized_keypoints):
    ''' normalize keypoints to image coordinates '''
    assert len(image_shape) in [2,3]
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape

    keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
    return keypoints

def put_on_device(items, device):
    for i, item in enumerate(items):
        items[i] = item.to(device)
    return items

def draw_kp(img, keypoints, name=None, normalized=True, RGB=True, num_keypoints=9, label=None):
    '''
    img: numpy three dimensional array
    keypoints: array like with shape [9,2]
    name: path to save
    '''
    img_copy = img.copy()
    # if image transposed
    if img_copy.shape[0] == 3:
        img_copy = np.transpose(img_copy, (1,2,0))
    # if image in RGB space --> convert to BGR
    img_copy = cv.cvtColor(img_copy, cv.COLOR_RGB2BGR) if RGB else img_copy
    # expand dim with zeros, needed for drawing function API
    expanded_kp = np.zeros((num_keypoints,3))
    keypoints = keypoints if normalized else normalize(img_copy.shape, keypoints)
    expanded_kp[:,:2] = keypoints
    graphics.draw_annotation_on_image(img_copy, expanded_kp , [num_keypoints])
    # put class label if given
    if label:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_copy, str(label), (10,180), font, 1, (0, 255, 0), 2, cv.LINE_AA)
    if name:
        cv.imwrite(name, img_copy)
    return img_copy

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    """Writes console output to external text file.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_
    Args:
        fpath (str): directory to save logging file.
    Examples::
       >>> import sys
       >>> import os
       >>> import os.path as osp
       >>> save_dir = 'log/resnet50-softmax-market1501'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
