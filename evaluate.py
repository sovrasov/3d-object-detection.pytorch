import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
module_path = os.path.abspath(os.path.join('/home/prokofiev/3D-object-recognition/3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

from metrics import compute_average_distance
from models import mobilenetv3_large

def evaluate(model, test_loader):
    print("Not emplemented yet")
    pass
