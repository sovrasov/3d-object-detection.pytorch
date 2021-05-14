import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite_pytorch.utils import get_model_params
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
from efficientnet_lite1_pytorch_model import EfficientnetLite1ModelFile
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile

from torchdet3d.models import (MobileNetV3, init_pretrained_weights,
                               model_params, MobileNetV3_large_100_timm)
from torchdet3d.utils import load_pretrained_weights

__AVAI_MODELS__ = {
                    'mobilenetv3_large', 'mobilenetv3_small', 'efficientnet-lite0', 'efficientnet-lite1',
                    'efficientnet-lite2', 'mobilenetv3_large_21k',
                  }

EFFICIENT_NET_WEIGHTS = {
                         'efficientnet-lite0' : EfficientnetLite0ModelFile.get_model_file_path(),
                         'efficientnet-lite1' : EfficientnetLite1ModelFile.get_model_file_path(),
                         'efficientnet-lite2' : EfficientnetLite2ModelFile.get_model_file_path()
                         }

def build_model(config, export_mode=False, weights_path=''):
    assert config.model.name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if config.model.name.startswith('efficientnet'):
        weights_path = EFFICIENT_NET_WEIGHTS[config.model.name]
        blocks_args, global_params = get_model_params(config.model.name, override_params=None)
        model = model_wrapper(model_class=EfficientNet,
                             output_channels=1280,
                             num_classes=config.model.num_classes,
                             blocks_args=blocks_args,
                             global_params=global_params)

        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)
        elif config.model.pretrained and not export_mode:
            load_pretrained_weights(model, weights_path)

    elif config.model.name == 'mobilenetv3_large':
        params = model_params[config.model.name]
        model = model_wrapper(model_class=MobileNetV3, output_channels=1280,
                                num_classes=config.model.num_classes, export_mode=export_mode, **params)

        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)
        elif config.model.pretrained and not export_mode:
            init_pretrained_weights(model, key=config.model.name)

    elif config.model.name == 'mobilenetv3_small':
        params = model_params[config.model.name]
        model = model_wrapper(model_class=MobileNetV3, output_channels=1024,
                                num_classes=config.model.num_classes, export_mode=export_mode, **params)

        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)
        elif config.model.pretrained and not export_mode:
            init_pretrained_weights(model, key=config.model.name)

    elif config.model.name == 'mobilenetv3_large_21k':
        model = model_wrapper(model_class=MobileNetV3_large_100_timm, output_channels=1280,
                             num_classes=config.model.num_classes, export_mode=export_mode)

        if config.model.load_weights:
            load_pretrained_weights(model, config.model.load_weights)
        elif config.model.pretrained and not export_mode:
            init_pretrained_weights(model, key=config.model.name, extra_prefix='model.')

    return model

def model_wrapper(model_class, output_channels, num_points=18,
                    num_classes=1, pooling_mode='avg', export_mode=False, **kwargs):
    class ModelWrapper(model_class):
        def __init__(self, output_channel=output_channels, **kwargs):
            super().__init__(**kwargs)
            max_classes = 9
            self.regressors = nn.ModuleList()
            for _ in range(max_classes):
                self.regressors.append(self._init_regressors(output_channel))
            self.cls_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(output_channel, num_classes),
            )

            self.sigmoid = nn.Sigmoid()

        @staticmethod
        def _init_regressors(output_channel=1280):
            return nn.Sequential(
                nn.Linear(output_channel, num_points),
            )

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
            if model_class is MobileNetV3:
                pooled_features = self.classifier(pooled_features)
            predicted_output = list()
            for reg in self.regressors:
                predicted_output.append(reg(pooled_features).view(1, x.size(0), num_points // 2, 2))
            predicted_output = self.sigmoid(torch.cat(predicted_output))
            predicted_targets = self.cls_fc(pooled_features) if num_classes > 1 else torch.zeros(x.size(0))
            return predicted_output, predicted_targets

        def forward(self, x, cats):
            ''' ordinary forward for training '''
            features = self.extract_features(x)
            pooled_features = self._glob_feature_vector(features, mode=pooling_mode)
            if model_class is MobileNetV3:
                pooled_features = self.classifier(pooled_features)
            # unique_cats = torch.unique(cats)
            # separated_features = [pooled_features[cats==cls] for cls in unique_cats]
            # assert len(separated_features) == len(unique_cats)
            # kp = torch.cat([self.regressors[c](minibatch) for c, minibatch in
            #                         zip(unique_cats, separated_features)])
            kp = torch.cat([self.regressors[c](sample) for c, sample in zip(cats, pooled_features)])
            kp = self.sigmoid(kp)
            kp = kp.view(x.size(0), num_points // 2, 2)
            # kp = kp[torch.randperm(kp.size(0))] # shuffle data
            if num_classes > 1:
                targets = self.cls_fc(pooled_features)
            else:
                targets = cats.unsqueeze(dim=1)

            return kp, targets

    model = ModelWrapper(**kwargs)
    if export_mode:
        model.forward = model.forward_to_onnx
    return model
