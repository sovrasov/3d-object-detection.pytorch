from __future__ import absolute_import, print_function

import sys
import os

module_path = os.path.abspath(os.path.join(os.path.dirname('__init__.py'), '3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

#pylint: disable = wrong-import-position
from torchdet3d import builders, evaluation, dataloaders, trainer, models, utils, losses
from .version import __version__

__author__ = 'Sovrasov Vladislav, Prokofiev Kirill'
__description__ = 'A library for deep learning 3D object detection in PyTorch'
__url__ = 'https://github.com/sovrasov/3d-object-detection.pytorch'
