import torch

def compute_average_distance(pred_box, gt_box, num_keypoint=9):
    """Computes Average Distance (ADD) metric."""
    detached_pred_box = pred_box.detach()
    detached_gt_box = gt_box.detach()
    add_distance = 0.
    # compute
    add_distance = torch.mean(torch.linalg.norm(detached_pred_box - detached_gt_box, dim=2))

    # Computes the symmetric version of the average distance metric.
    add_sym_distance = torch.zeros((detached_pred_box.shape[0])).to(detached_pred_box.device)
    for i in range(num_keypoint):
        # Find nearest vertex in gt_box
        distance = torch.linalg.norm(detached_pred_box[:, i, :] - detached_gt_box[:, i, :], dim=1)
        for j in range(num_keypoint):
            d = torch.linalg.norm(detached_pred_box[:, i, :] - detached_gt_box[:, j, :], dim=1)
            distance = torch.where(d < distance, d, distance)
        add_sym_distance += distance

    # average by num keypoints, then mean by batch
    add_sym_distance = torch.mean(add_sym_distance / num_keypoint)
    return add_distance.item(), add_sym_distance.item()

def compute_accuracy(pred_cats, gt_cats):
    detached_pred_cats = pred_cats.detach()
    detached_gt_cats = gt_cats.detach() if isinstance(gt_cats, torch.Tensor) else gt_cats
    detached_pred_cats = torch.argmax(detached_pred_cats, dim=1)
    return torch.mean((detached_pred_cats == detached_gt_cats).float()).item()
