import pickle
import sys
import os
import errno

from collections import OrderedDict
import os.path as osp
from functools import partial
from importlib import import_module

import numpy as np
import cv2 as cv
import torch
import warnings
from attrdict import AttrDict as adict

from objectron.dataset import graphics

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

def save_snap(model, optimizer, epoch, log_path):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
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
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
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

def load_pretrained_weights(model, file_path='', pretrained_dict=None, resume=False):
    r"""Loads pretrianed weights to model. Imported from openvinotoolkit/deep-object-reid.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        file_path (str): path to pretrained weights.
    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> file_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, file_path)
    """
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
        if k.startswith('module.') and not resume:
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(file_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(file_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def unnormalize(image, normalized_keypoints):
    ''' transform image to global pixel values '''
    h, w, _ = image.shape
    keypoints = np.multiply(normalized_keypoints, np.asarray([w, h], np.float32)).astype(int)
    return keypoints

def normalize(image, unnormalized_keypoints):
    ''' normalize keypoints to image coordinates '''
    h, w, _ = image.shape
    keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
    return keypoints

def draw_kp(img, keypoints, name, normalized=True, RGB=True, num_keypoints=9):
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
    keypoints = keypoints if normalized else normalize(img_copy, keypoints)
    expanded_kp[:,:2] = keypoints
    graphics.draw_annotation_on_image(img_copy, expanded_kp , [num_keypoints])
    cv.imwrite(name, img_copy)

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
