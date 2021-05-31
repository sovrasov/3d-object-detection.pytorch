import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import numpy as np


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_joints, cost_class: float = 1, cost_coord: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coord: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.num_joints = num_joints
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        ## target: [bs, 17, 2]
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_kpt = outputs["pred_coords"]  # [batch_size, num_queries, 2]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[..., :self.num_joints]

        # Compute the L1 cost between keypoints
        cost_kpt = torch.cdist(out_kpt, targets, p=1)  # [B, N, 17]

        # Final cost matrix
        C = self.cost_coord * cost_kpt + self.cost_class * cost_class
        C = C.transpose(1, 2).cpu()  # [B, 17, N]

        indices = [linear_sum_assignment(c) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(num_joints, cost_class=1.0, cost_coord=5.0):
    return HungarianMatcher(num_joints, cost_class=cost_class, cost_coord=cost_coord)


def get_final_preds_match(outputs):
    pred_logits = outputs['pred_logits'].detach()
    pred_coords = outputs['pred_coords'].detach()

    num_joints = pred_logits.shape[-1] - 1

    # exclude background logit
    prob = F.softmax(pred_logits[..., :-1], dim=-1)

    score_holder = []
    orig_coord = []
    for b, C in enumerate(prob):
        # Cost Matrix: [num_joints, N]
        _, query_ind = linear_sum_assignment(-C.cpu().numpy().transpose(0, 1))
        score = prob[b, query_ind, list(np.arange(num_joints))][..., None]
        pred_raw = pred_coords[b, query_ind]
        orig_coord.append(pred_raw)
        score_holder.append(score)

    matched_score = torch.stack(score_holder)
    matched_coord = torch.stack(orig_coord)

    return matched_coord, matched_score


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    @torch.no_grad()
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes_o = src_idx[1].to(src_logits.device)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # default to no-kpt class, for matched ones, set to 0, ..., 16
        target_classes[tgt_idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - self.accuracy(src_logits[tgt_idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = pred_logits.new_ones(pred_logits.shape[0]) * self.num_classes
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_kpts(self, outputs, targets, indices, weights):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_coords' in outputs
        # match gt --> pred
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # must be in range(0, 100)

        src_kpts = targets[src_idx]
        weights = weights[src_idx]

        target_kpts = outputs['pred_coords'][tgt_idx]

        loss_bbox = F.l1_loss(src_kpts, target_kpts, reduction='none') * weights

        losses = {'loss_kpts': loss_bbox.mean() * self.num_classes}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'kpts': self.loss_kpts,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, target_weights, predictions_only=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        idx = self._get_tgt_permutation_idx(indices)
        src_kpts = outputs['pred_coords'][idx]
        pred = (src_kpts.view(-1, self.num_classes, 2) * target_weights).detach()
        if predictions_only:
            return pred

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'kpts':
                losses.update(self.get_loss(loss, outputs, targets, indices, weights=target_weights))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    elif loss == 'kpts':
                        kwargs = {'weights': target_weights}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses, pred


class PRTRLossWrapper(nn.Module):
    def __init__(self, set_criterion, img_size, num_points):
        super().__init__()
        self.set_criterion = set_criterion
        self.num_points = num_points

    def forward(self, outputs, target):
        target_weight = torch.ones((target.shape[0], self.num_points, 1)).to(target.device)
        loss_dict, _ = self.set_criterion(outputs, target, target_weight)
        weight_dict = self.set_criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys() if k in weight_dict)
        return loss

    def get_val_predictions(self, outputs, target):
        target_weight = torch.ones((target.shape[0], self.num_points, 1)).to(target.device)
        predictions = self.set_criterion(outputs, target, target_weight, predictions_only=True)
        return predictions


def build_prtr_loss(num_points=9, img_size=(224, 224)):
    matcher = build_matcher(num_points)
    weight_dict = {'loss_ce': 1.0, 'loss_kpts': 5.0}

    aux_weight_dict = {}
    decodel_layers = 6
    for i in range(decodel_layers - 1):
        aux_weight_dict.update(
            {k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    criterion = SetCriterion(num_points, matcher, weight_dict, 0.1, [
        'labels', 'kpts', 'cardinality']).cuda()

    return PRTRLossWrapper(criterion, img_size, num_points)
