import numpy as np

from torchdet3d.utils import (lift_2d, get_default_camera_matrix,
                              convert_camera_matrix_2_ndc, project_3d_points,
                              convert_2d_to_ndc)


from objectron.dataset import iou
from objectron.dataset import box


class TestCasesGeometry:
    test_kps = np.array([[0.47714591, 0.47491544],
                         [0.73884577, 0.39749265],
                         [0.18508956, 0.40002537],
                         [0.74114597, 0.48664019],
                         [0.18273196, 0.48833901 ],
                         [0.64639187, 0.46719882],
                         [0.32766378, 0.46827659],
                         [0.64726073, 0.51853681],
                         [0.32699507, 0.51933688]])
    EPS = 1e-5
    IOU_THR = 0.5

    def test_reprojection_error(self):
        kps_3d = lift_2d([self.test_kps], portrait=True)[0]
        reprojected_kps = project_3d_points(kps_3d, convert_camera_matrix_2_ndc(get_default_camera_matrix()))
        test_kps_ndc = convert_2d_to_ndc(self.test_kps, portrait=True)
        assert np.any(np.linalg.norm(test_kps_ndc - reprojected_kps, axis=1) < self.EPS)

    def test_3d_iou_stability(self):
        np.random.seed(10)
        noisy_kps = np.clip(self.test_kps + 0.01*np.random.rand(*self.test_kps.shape), 0, 1)
        lifted_3d_sets = lift_2d([self.test_kps, noisy_kps], portrait=True)

        b1 = box.Box(vertices=lifted_3d_sets[0])
        b2 = box.Box(vertices=lifted_3d_sets[1])

        loss = iou.IoU(b1, b2)
        assert loss.iou() > self.IOU_THR
