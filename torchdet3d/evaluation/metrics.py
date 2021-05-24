import torch
import scipy
import numpy as np

from objectron.dataset import iou
from objectron.dataset import box

from torchdet3d.utils import lift_2d

@torch.no_grad()
def compute_average_distance(pred_kp, gt_kp, num_keypoint=9, reduce_mean=True, **kwargs):
    """Computes Average Distance (ADD) metric."""
    # Computes the symmetric version of the average distance metric.
    add_sym_distance = torch.zeros((pred_kp.shape[0])).to(pred_kp.device)
    for i in range(num_keypoint):
        # Find nearest vertex in gt_kp
        distance = torch.linalg.norm(pred_kp[:, i, :] - gt_kp[:, i, :], dim=1)
        for j in range(num_keypoint):
            d = torch.linalg.norm(pred_kp[:, i, :] - gt_kp[:, j, :], dim=1)
            distance = torch.where(d < distance, d, distance)
        add_sym_distance += distance

    if reduce_mean:
        add_distance = torch.mean(torch.linalg.norm(pred_kp - gt_kp, dim=2))
        add_sym_distance = torch.mean(add_sym_distance) / num_keypoint
    else:
        add_distance = torch.sum(torch.linalg.norm(pred_kp - gt_kp, dim=2)) / num_keypoint
        add_sym_distance = torch.sum(add_sym_distance) / num_keypoint
    return add_distance.item(), add_sym_distance.item()

@torch.no_grad()
def compute_accuracy(pred_cats, gt_cats, reduce_mean=True, **kwargs):
    pred_cats = torch.argmax(pred_cats, dim=1)
    if reduce_mean:
        return torch.mean((pred_cats == gt_cats).float()).item()

    return torch.sum((pred_cats == gt_cats).float()).item()

@torch.no_grad()
def compute_metrics_per_cls(pred_kp, gt_kp, pred_cats, gt_cats, compute_iou=True, **kwargs):
    classes = torch.unique(gt_cats)
    computed_metrics = []
    total_ADD, total_SADD, total_IOU, total_acc = 0, 0, 0, 0
    batch_size = pred_kp.shape[0]
    for cl in classes:
        class_gt_kp = gt_kp[gt_cats == cl]
        class_pred_kp = pred_kp[gt_cats == cl]
        ADD, SADD = compute_average_distance(class_pred_kp,
                                             class_gt_kp, reduce_mean=False,
                                             **kwargs)
        if compute_iou:
            IOU = compute_2d_based_iou(class_pred_kp,
                                       class_gt_kp, reduce_mean=False)
        else:
            IOU = 0.
        acc = compute_accuracy(pred_cats[gt_cats == cl],
                               gt_cats[gt_cats == cl], reduce_mean=False,
                               **kwargs)
        num_class_samples = class_gt_kp.shape[0]
        computed_metrics.append((cl, ADD / num_class_samples, SADD / num_class_samples,
                                 IOU / num_class_samples, acc / num_class_samples))
        total_ADD += ADD
        total_SADD += SADD
        total_IOU += IOU
        total_acc += acc

    return (computed_metrics, total_ADD / batch_size, total_SADD / batch_size,
            total_IOU / batch_size, total_acc / batch_size)

@torch.no_grad()
def compute_2d_based_iou(pred_kp: torch.Tensor, gt_kp: torch.Tensor, reduce_mean: bool=True):
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
    if reduce_mean:
        return total_iou / bs if bs else 0
    return total_iou
