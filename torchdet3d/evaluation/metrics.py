import torch
import scipy
import numpy as np

from objectron.dataset import iou
from objectron.dataset import box

from torchdet3d.utils import lift_2d


def compute_average_distance(pred_kp, gt_kp, num_keypoint=9, **kwargs):
    """Computes Average Distance (ADD) metric."""
    detached_pred_kp = pred_kp.detach()
    detached_gt_kp = gt_kp.detach()
    add_distance = 0.
    # compute
    add_distance = torch.mean(torch.linalg.norm(detached_pred_kp - detached_gt_kp, dim=2))

    # Computes the symmetric version of the average distance metric.
    add_sym_distance = torch.zeros((detached_pred_kp.shape[0])).to(detached_pred_kp.device)
    for i in range(num_keypoint):
        # Find nearest vertex in gt_kp
        distance = torch.linalg.norm(detached_pred_kp[:, i, :] - detached_gt_kp[:, i, :], dim=1)
        for j in range(num_keypoint):
            d = torch.linalg.norm(detached_pred_kp[:, i, :] - detached_gt_kp[:, j, :], dim=1)
            distance = torch.where(d < distance, d, distance)
        add_sym_distance += distance

    # average by num keypoints, then mean by batch
    add_sym_distance = torch.mean(add_sym_distance / num_keypoint)
    return add_distance.item(), add_sym_distance.item()

def compute_accuracy(pred_cats, gt_cats, **kwargs):
    detached_pred_cats = pred_cats.detach()
    detached_gt_cats = gt_cats.detach() if isinstance(gt_cats, torch.Tensor) else gt_cats
    detached_pred_cats = torch.argmax(detached_pred_cats, dim=1)
    return torch.mean((detached_pred_cats == detached_gt_cats).float()).item()

def compute_metrics_per_cls(pred_kp, gt_kp, gt_cats, pred_cats, **kwargs):
    classes = torch.unique(gt_cats)
    computed_metrics = []
    for cl in classes:
        ADD, SADD = compute_average_distance(pred_kp[gt_cats == cl],
                                             gt_kp[gt_cats == cl],
                                             **kwargs)
        acc = compute_accuracy(pred_cats[gt_cats == cl],
                               gt_cats[gt_cats == cl],
                               **kwargs)
        computed_metrics.append((cl, ADD, SADD, acc))

    return computed_metrics

def compute_2d_based_iou(pred_kp: torch.Tensor, gt_kp: torch.Tensor):
    assert len(pred_kp.shape) == 3
    bs = pred_kp.shape[0]
    pred_kp_np = pred_kp.cpu().numpy()
    gt_kp_np = gt_kp.cpu().numpy()
    total_iou = 0
    for i in range(bs):
        kps_3d = lift_2d([pred_kp_np[i], gt_kp_np[i]], portrait=True)
        b_pred = box.Box(vertices=kps_3d[0])
        b_gt = box.Box(vertices=kps_3d[1])
        try:
            total_iou += iou.IoU(b_pred, b_gt).iou()
        except scipy.spatial.qhull.QhullError:
            pass
        except np.linalg.LinAlgError:
            pass
    return total_iou / bs
