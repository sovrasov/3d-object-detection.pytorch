import argparse

import torch
from icecream import ic
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite_pytorch.utils import get_model_params
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
from efficientnet_lite1_pytorch_model import EfficientnetLite1ModelFile
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile

from torchdet3d.models import MobileNetV3, init_pretrained_weights, model_params
from torchdet3d.utils import load_pretrained_weights

__AVAI_MODELS__ = {
                    'mobilenetv3_large', 'mobilenetv3_small', 'efficientnet-lite0', 'efficientnet-lite1',
                    'efficientnet-lite2',
                  }

EFFICIENT_NET_WEIGHTS = {
                         'efficientnet-lite0' : EfficientnetLite0ModelFile.get_model_file_path(),
                         'efficientnet-lite1' : EfficientnetLite1ModelFile.get_model_file_path(),
                         'efficientnet-lite2' : EfficientnetLite2ModelFile.get_model_file_path()
                         }

def build_model(config, export_mode=False):
    assert config.model.name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if config.model.name.startswith('efficientnet'):
        weights_path = EFFICIENT_NET_WEIGHTS[config.model.name]
        blocks_args, global_params = get_model_params(config.model.name, override_params=None)
        model = model_wraper(model_class=EfficientNet,
                             output_channels=1280,
                             num_classes=config.model.num_classes,
                             blocks_args=blocks_args,
                             global_params=global_params)

        if config.model.pretrained and not export_mode:
            load_pretrained_weights(model, weights_path)
        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)

    elif config.model.name == 'mobilenetv3_large':
        params = model_params['mobilenetv3_large']
        model = model_wraper(model_class=MobileNetV3, output_channels=1280,
                                num_classes=config.model.num_classes, export_mode=export_mode, **params)
        if config.model.pretrained and not export_mode:
            init_pretrained_weights(model, key='mobilenetv3_large')
        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)

    elif config.model.name == 'mobilenetv3_small':
        params = model_params['mobilenetv3_small']
        model = model_wraper(model_class=MobileNetV3, output_channels=1024,
                                num_classes=config.model.num_classes, export_mode=export_mode, **params)
        if config.model.pretrained and not export_mode:
            init_pretrained_weights(model, key='mobilenetv3_small')
        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)

    return model

def model_wraper(model_class, output_channels, num_points=18,
                    num_classes=1, pooling_mode='avg', export_mode=False, **kwargs):
    class ModelWrapper(model_class):
        def __init__(self, output_channel=output_channels, **kwargs):
            super().__init__(**kwargs)
            self.regressors = nn.ModuleList()
            for _ in range(num_classes):
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

        def forward_to_onnx(self, x):
            ''' use trained classificator to predict object's class and
                choose according head for this '''
            features = self.extract_features(x)
            pooled_features = self._glob_feature_vector(features, mode=pooling_mode)
            predicted_output = list()
            if len(self.regressors) > 1:
                for reg in self.regressors[1:]:
                    predicted_output.append(reg(pooled_features).view(1, x.size(0), num_points // 2, 2))
            predicted_output = self.sigmoid(torch.cat(predicted_output))

            return predicted_output

        def forward(self, x, cats):
            ''' ordinary forward for training '''
            features = self.extract_features(x)
            pooled_features = self._glob_feature_vector(features, mode=pooling_mode)
            kp = torch.cat([self.regressors[id_](sample) for id_, sample in zip(cats, pooled_features)], 0)
            kp = self.sigmoid(kp)
            if num_classes > 1:
                targets = self.classifier(pooled_features)
            else:
                targets = cats.unsqueeze(dim=1)

            return (kp.view(x.size(0), num_points // 2, 2), targets)

    model = ModelWrapper(**kwargs)
    if export_mode:
        model.forward = model.forward_to_onnx
    return model

def test(config):
    model = build_model(config)
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
