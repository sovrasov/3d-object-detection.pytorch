from typing import List, Tuple

import numpy as np


__epnp_alpha__ = np.array([[4, -1, -1, -1],
                           [2, -1, -1,  1],
                           [2, -1,  1, -1],
                           [0, -1,  1,  1],
                           [2,  1, -1, -1],
                           [0,  1, -1,  1],
                           [0,  1,  1, -1],
                           [-2, 1,  1,  1]])


def get_default_camera_matrix():
    return np.array([[1, 0, 0.5],
                     [0, 1, 0.5],
                     [0, 0,  1]])


def project_3d_points(points: np.array, camera_matrix: np.array):
    assert len(points.shape) == 2
    projection = np.matmul(camera_matrix, points.T).T
    projection /= -projection[:,2].reshape(-1, 1)
    return projection[:, :-1]


def convert_camera_matrix_2_ndc(matrix: np.array, img_shape: Tuple[int, int]=(1, 1)):
    ndc_mat = np.copy(matrix)
    ndc_mat[0, 0] *= 2.0 / img_shape[0]
    ndc_mat[1, 1] *= 2.0 / img_shape[1]

    ndc_mat[0, 2] = -ndc_mat[0, 2] * 2.0 / img_shape[0]  + 1.0
    ndc_mat[1, 2] = -ndc_mat[1, 2] * 2.0 / img_shape[1]  + 1.0

    return ndc_mat


def convert_2d_to_ndc(points: np.array, portrait: bool=False):
    converted_points = np.zeros_like(points)
    if portrait:
        converted_points[:, 0] = points[:, 1] * 2 - 1
        converted_points[:, 1] = points[:, 0] * 2 - 1
    else:
        converted_points[:, 0] = points[:, 0] * 2 - 1
        converted_points[:, 1] = 1 - points[:, 1] * 2
    return converted_points


def lift_2d(keypoint_sets: List[np.array],
            camera_matrix: np.array=get_default_camera_matrix(),
            portrait: bool=False) -> List[np.array]:
    """
    Function takes normalized 2d coordinates of 2d keypoints on the image plane,
    camera matrix in normalized image space and outputs lifted 3d points in camera coordinates,
    which are defined up to an unknown scale factor
    """
    ndc_cam_mat = convert_camera_matrix_2_ndc(camera_matrix)
    fx = ndc_cam_mat[0, 0]
    fy = ndc_cam_mat[1, 1]
    cx = ndc_cam_mat[0, 2]
    cy = ndc_cam_mat[1, 2]

    lifted_keypoint_sets = []

    for kp_set in keypoint_sets:
        m = np.zeros((16, 12))
        assert len(kp_set) == 9

        for i in range(8):
            kp = kp_set[i + 1]
            # Convert 2d point from normalized screen coordinates [0, 1] to NDC coordinates([-1, 1]).
            if portrait:
                u = kp[1] * 2 - 1
                v = kp[0] * 2 - 1
            else:
                u = kp[0] * 2 - 1
                v = 1 - kp[1] * 2

            for j in range(4):
                # For each of the 4 control points, formulate two rows of the
                # m matrix (two equations).
                control_alpha = __epnp_alpha__[i, j]
                m[i * 2, j * 3] = fx * control_alpha
                m[i * 2, j * 3 + 2] = (cx + u) * control_alpha
                m[i * 2 + 1, j * 3 + 1] = fy * control_alpha
                m[i * 2 + 1, j * 3 + 2] = (cy + v) * control_alpha

        mt_m = np.matmul(m.T, m)
        w, v = np.linalg.eigh(mt_m)
        assert w.shape[0] == 12
        control_matrix = v[:, 0].reshape(4, 3)
        # All 3d points should be in front of camera (z < 0).
        if control_matrix[0, 2] > 0:
            control_matrix = -control_matrix

        lifted_kp_set = []
        lifted_kp_set.append(control_matrix[0, :])
        vertices = np.matmul(__epnp_alpha__, control_matrix)

        for i in range(8):
            lifted_kp_set.append(vertices[i, :])

        lifted_kp_set = np.array(lifted_kp_set)
        lifted_keypoint_sets.append(lifted_kp_set)

    return lifted_keypoint_sets


def draw_boxes(boxes=[], clips=[], colors=['r', 'b', 'g', 'k']):
    """Draw a list of boxes.
        The boxes are defined as a list of vertices
    """
    import matplotlib.pyplot as plt
    from objectron.dataset import box

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, b in enumerate(boxes):
        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        ax.scatter(x, y, z, c='r')
        for e in box.EDGES:
            ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

    if clips:
        points = np.array(clips)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    # rotate the axes and update
    ax.view_init(30, 12)
    plt.draw()
    plt.savefig('3d_boxes.png')
