import argparse

import torch
from icecream import ic
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite_pytorch.utils import get_model_params
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

from torchdet3d.models import mobilenetv3_large, MobileNetV3, init_pretrained_weights, model_params
from  torchdet3d.utils import load_pretrained_weights

def build_model(cfg):
    if cfg.model.name.startswith('efficientnet'):
        weights_path = EfficientnetLite0ModelFile.get_model_file_path()
        blocks_args, global_params = get_model_params(cfg.model.name, override_params=None)
        model = model_wraper(model_class=EfficientNet,
                             num_classes=cfg.model.num_classes,
                             blocks_args=blocks_args,
                             global_params=global_params)

        if cfg.model.pretrained:
            load_pretrained_weights(model, weights_path)

    elif cfg.model.name == 'mobilenetv3_large':
            params = model_params['mobilenetv3_large']
            model = model_wraper(model_class=MobileNetV3, num_classes=cfg.model.num_classes, **params)
            if cfg.model.pretrained:
                init_pretrained_weights(model, key='mobilenetv3_large')

    elif cfg.model.name == 'mobilenetv3_small':
            params = model_params['mobilenetv3_small']
            model = model_wraper(model_class=MobileNetV3, num_classes=cfg.model.num_classes, **params)
            if cfg.model.pretrained:
                init_pretrained_weights(model, key='mobilenetv3_small')

    return model

def model_wraper(model_class, num_points=18, num_classes=1, pooling_mode='avg', **kwargs):
    class ModelWrapper(model_class):
        def __init__(self, output_channel=1280,**kwargs):
            super().__init__(**kwargs)
            self.regressors = nn.ModuleList()
            for trg_id, trg_num_classes in enumerate(range(num_classes)):
                self.regressors.append(self._init_regressors(output_channel))
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(output_channel, num_classes),
            )

            self.sigmoid = nn.Sigmoid()

        @staticmethod
        def _init_regressors(output_channel):
            return nn.Linear(output_channel, num_points)

        @staticmethod
        def _glob_feature_vector(x, mode, reduce_dims=True):
            if mode == 'avg':
                out = F.adaptive_avg_pool2d(x, 1)
            elif mode == 'max':
                out = F.adaptive_max_pool2d(x, 1)
            elif mode == 'avg+max':
                avg_pool = F.adaptive_avg_pool2d(x, 1)
                max_pool = F.adaptive_max_pool2d(x, 1)
                out = avg_pool + max_pool
            else:
                raise ValueError(f'Unknown pooling mode: {mode}')

            if reduce_dims:
                return out.view(x.size(0), -1)
            return out

        def forward(self, x, cats):
            features = self.extract_features(x)
            pooled_features = self._glob_feature_vector(features, mode=pooling_mode)
            ic(pooled_features.shape, cats.shape)
            kp = torch.cat([self.regressors[id](sample) for id, sample in zip(cats, pooled_features)], 0)
            kp = self.sigmoid(kp)
            targets = self.classifier(pooled_features) if num_classes > 1 else torch.zeros(kp.size(0), num_classes).to(x.device)
            return kp.view(x.size(0), num_points // 2, 2), targets

    model = ModelWrapper(**kwargs)
    return model

def test(cfg):
    model = build_model(cfg)
    img = torch.rand(128,3,224,224)
    cats = torch.randint(0,5,(128,))
    out = model(img, cats)
    ic(out[0].shape, out[1].shape)

if __name__ == "__main__":
    from torchdet3d.utils import read_py_config
    parser = argparse.ArgumentParser(description='3D-object-detection training')
    parser.add_argument('--config', type=str, default='./configs/debug_config.py', help='path to config')
    args = parser.parse_args()
    cfg = read_py_config(args.config)
    test(cfg)
