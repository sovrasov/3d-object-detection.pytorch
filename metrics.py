import torch


def compute_average_distance(pred_box, gt_box, num_keypoint=9):
    """Computes Average Distance (ADD) metric."""
    detached_pred_box = pred_box.detach()
    detached_gt_box = gt_box.detach()
    add_distance = 0.
    for i in range(num_keypoint):
        delta = torch.linalg.norm(detached_pred_box[:, i, :] - detached_gt_box[:, i, :])
        add_distance += delta
    add_distance /= num_keypoint

    # Computes the symmetric version of the average distance metric.
    add_sym_distance = 0.
    for i in range(num_keypoint):
        # Find nearest vertex in gt_box
        distance = torch.linalg.norm(detached_pred_box[:, i, :] - detached_gt_box[:, i, :])
        for j in range(num_keypoint):
            d = torch.linalg.norm(detached_pred_box[:, i, :] - detached_gt_box[:, j, :])
            if d < distance:
                distance = d
        add_sym_distance += distance

    add_sym_distance /= num_keypoint

    return add_distance, add_sym_distance
