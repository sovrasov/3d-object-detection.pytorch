import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mobilenetv3 import mobilenetv3_large_100

from torchdet3d.models.transformer import build_transformer


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PoseTransformer(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_points):
        super(PoseTransformer, self).__init__()
        self.num_queries = 100
        self.num_classes = num_classes
        self.num_points = num_points
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, self.num_points + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(1280, hidden_dim, 1)
        self.aux_loss = True

        self.cls_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.backbone.num_channels[0], num_classes),
            )

    def forward(self, x, cats=None):
        src, pos = self.backbone(x)
        hs = self.transformer(self.input_proj(src[-1]), None,
                              self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()

        if self.num_classes > 1:
            targets = self.cls_fc(src[0].view(src[0].shape[0], -1))
        else:
            targets = cats.unsqueeze(dim=1)

        out = {'pred_logits': outputs_class[-1],
               'pred_coords': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out, targets

    def forward_to_onnx(self, x):
        src, pos = self.backbone(x)
        hs = self.transformer(self.input_proj(src[-1]), None,
                              self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()

        predicted_targets = self.cls_fc(src) if self.num_classes > 1 else torch.zeros(x.size(0))
        return outputs_class[-1], outputs_coord[-1], predicted_targets

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, _, h, w = x.shape
        not_mask = torch.ones(bs, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: torch.Tensor):
        out = self[0](tensor_list)
        pos = []

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.dtype))

        return out, pos


class MobilenetV3Backbone(nn.Module):
    def __init__(self, return_interm_layers: bool = False, pretrained: bool = False):
        super().__init__()
        if return_interm_layers:
            raise NotImplementedError(
                "MobileNetV3 backbone does not support return interm layers")
        else:
            self.num_channels = [1280]
        self.body = mobilenetv3_large_100(pretrained)
        self.body.classifier = None

    def forward(self, x):
        y = self.body.forward_features(x)
        return [y]


def get_mobilenetv3_pose_net(num_classes=9, pretrained=False, num_points=9, export_mode=False):
    hd = 256
    transformer = build_transformer(hidden_dim=hd, dropout=0.0, nheads=8, dim_feedforward=2048,
                                    enc_layers=6, dec_layers=6, pre_norm=False)
    n_steps = hd // 2
    position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
    backbone = MobilenetV3Backbone(pretrained=pretrained)
    model = Joiner(backbone, position_embedding)
    model = PoseTransformer(model, transformer, num_classes, num_points)
    if export_mode:
        model.forward = model.forward_to_onnx

    return model
