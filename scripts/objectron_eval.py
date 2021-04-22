"""Example Evaluation script for Objectron dataset.

It reads a tfrecord, runs evaluation, and outputs a summary report with name
specified in report_file argument. When adopting this for your own model, you
have to implement the Evaluator.predict() function, which takes an image and produces
a 3D bounding box.

Example:
  python3 -m objectron.dataset.eval --eval_data=.../chair_test* --report_file=.../report.txt
"""

import os
import sys

from absl import app
from absl import flags

import glob
import numpy as np
import tensorflow as tf
import tqdm
import cv2 as cv

from openvino.inference_engine import IECore

from torchdet3d.builders import build_model
from torchdet3d.utils import (read_py_config, Detector, Regressor,
                              OBJECTRON_CLASSES, draw_kp, lift_2d)

from objectron.dataset.eval import Evaluator
from objectron.dataset.eval import FLAGS
import objectron.dataset.metrics as metrics


def draw_detections(frame, reg_detections, det_detections, reg_only=True):
    """Draws detections and labels"""
    for det_out, reg_out in zip(det_detections, reg_detections):
        left, top, right, bottom = det_out[0]
        kp = reg_out[0]
        label = reg_out[1]
        label = OBJECTRON_CLASSES[label]
        if not reg_only:
            cv.rectangle(frame, (left, top), (right, bottom),
                         (0, 255, 0), thickness=2)

        frame = draw_kp(frame, kp, None, RGB=False, normalized=False)
        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame


flags.DEFINE_string('regression_model', None,
                    'path to regression model config')
flags.DEFINE_string('detection_model', None, 'path to xml')


class Torchdet3dEvaluator(Evaluator):
    def __init__(self, detection_model, regression_model, correct_class_index, **kwargs):
        super().__init__(**kwargs)
        self.detection_model = detection_model
        self.regression_model = regression_model
        self.correct_class_index = correct_class_index

    def predict(self, images, batch_size):
        """
        Implement your own function/model to predict the box's 2D and 3D
        keypoint from the input images.
        Note that the predicted 3D bounding boxes are correct up to a scale.
        You can use the ground planes to re-scale your boxes if necessary.

        Returns:
            A list of list of boxes for objects in images in the batch. Each box is
            a tuple of (point_2d, point_3d) that includes the predicted 2D and 3D vertices.
        """
        all_boxes = []
        images *= 255
        for i in range(batch_size):
            self.detection_model.run_async(images[i])
            detections = self.detection_model.wait_and_grab()
            outputs = self.regression_model.get_detections(
                images[i], detections)
            # vis = draw_detections(images[i], outputs, detections, reg_only=False)
            # cv.imwrite('eval.png', vis)
            boxes = []
            for kps, label in outputs:
                kps[:, 0] /= images[i].shape[1]
                kps[:, 1] /= images[i].shape[0]
                kps_3d = lift_2d([kps], portrait=True)[0]
                boxes.append((kps, kps_3d))
            all_boxes.append(boxes)
        return all_boxes

    def evaluate(self, batch):
        """Evaluates a batch of serialized tf.Example protos."""
        images, labels, projs, planes = [], [], [], []
        for serialized in batch:
            example = tf.train.Example.FromString(serialized)
            image, label = self.encoder.parse_example(example)
            images.append(image)
            labels.append(label)
            proj, _ = self.encoder.parse_camera(example)
            projs.append(proj)
            plane = self.encoder.parse_plane(example)
            planes.append(plane)

        #pred = self.model.predict(np.asarray(images), batch_size=len(batch))
        results = self.predict(np.asarray(images), batch_size=len(batch))

        # Creating some fake results for testing as well as example of what the
        # the results should look like.
        # results = []
        # for label in labels:
        #  instances = label['2d_instance']
        #  instances_3d = label['3d_instance']
        #  boxes = []
        #  for i in range(len(instances)):
        #    point_2d = np.copy(instances[i])
        #    point_3d = np.copy(instances_3d[i])
        #    for j in range(9):
        #      # Translating the box in 3D, this will have a large impact on 3D IoU.
        #      point_3d[j] += np.array([0.01, 0.02, 0.5])
        #    boxes.append((point_2d, point_3d))
        #  results.append(boxes)

        for boxes, label, plane in zip(results, labels, planes):
            instances = label['2d_instance']
            instances_3d = label['3d_instance']
            visibilities = label['visibility']
            num_instances = 0
            for instance, instance_3d, visibility in zip(
                    instances, instances_3d, visibilities):
                if (visibility > self._vis_thresh and
                        self._is_visible(instance[0]) and instance_3d[0, 2] < 0):
                    num_instances += 1

            # We don't have negative examples in evaluation.
            if num_instances == 0:
                continue

            iou_hit_miss = metrics.HitMiss(self._iou_thresholds)
            azimuth_hit_miss = metrics.HitMiss(self._azimuth_thresholds)
            polar_hit_miss = metrics.HitMiss(self._polar_thresholds)
            pixel_hit_miss = metrics.HitMiss(self._pixel_thresholds)
            add_hit_miss = metrics.HitMiss(self._add_thresholds)
            adds_hit_miss = metrics.HitMiss(self._adds_thresholds)

            num_matched = 0
            for box in boxes:
                box_point_2d, box_point_3d = box
                index = self.match_box(box_point_2d, instances, visibilities)
                if index >= 0:
                    num_matched += 1
                    pixel_error = self.evaluate_2d(box_point_2d, instances[index])
                    # If you only compute the 3D bounding boxes from RGB images,
                    # your 3D keypoints may be upto scale. However the ground truth
                    # is at metric scale. There is a hack to re-scale your box using
                    # the ground planes (assuming your box is sitting on the ground).
                    # However many models learn to predict depths and scale correctly.
                    scale = self.compute_scale(box_point_3d, plane)
                    box_point_3d = box_point_3d * scale
                    azimuth_error, polar_error, iou, add, adds = self.evaluate_3d(
                        box_point_3d, instances_3d[index])
                else:
                    pixel_error = _MAX_PIXEL_ERROR
                    azimuth_error = _MAX_AZIMUTH_ERROR
                    polar_error = _MAX_POLAR_ERROR
                    iou = 0.
                    add = _MAX_DISTANCE
                    adds = _MAX_DISTANCE

                iou_hit_miss.record_hit_miss(iou)
                add_hit_miss.record_hit_miss(add, greater=False)
                adds_hit_miss.record_hit_miss(adds, greater=False)
                pixel_hit_miss.record_hit_miss(pixel_error, greater=False)
                azimuth_hit_miss.record_hit_miss(azimuth_error, greater=False)
                polar_hit_miss.record_hit_miss(polar_error, greater=False)

            self._iou_ap.append(iou_hit_miss, len(instances))
            self._pixel_ap.append(pixel_hit_miss, len(instances))
            self._azimuth_ap.append(azimuth_hit_miss, len(instances))
            self._polar_ap.append(polar_hit_miss, len(instances))
            self._add_ap.append(add_hit_miss, len(instances))
            self._adds_ap.append(adds_hit_miss, len(instances))
            self._matched += num_matched


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    class_name = FLAGS.eval_data.split(os.path.sep)[-2]
    correct_class_index = OBJECTRON_CLASSES.index(class_name)

    ie = IECore()
    object_detector = Detector(ie, FLAGS.detection_model, 0.5)
    regression_model = Regressor(ie, FLAGS.regression_model)

    print('Starting evaluation')
    evaluator = Torchdet3dEvaluator(
        object_detector, regression_model, correct_class_index)

    i = 0
    ds = tf.data.TFRecordDataset(
        glob.glob(FLAGS.eval_data)).take(FLAGS.max_num)
    batch = []
    for serialized in tqdm.tqdm(ds):
        batch.append(serialized.numpy())
        if len(batch) == FLAGS.batch_size:
            evaluator.evaluate(batch)
            batch.clear()
        i += 1
        if i > 4:
            break

    if batch:
        evaluator.evaluate(batch)

    evaluator.finalize()
    evaluator.write_report()


if __name__ == '__main__':
    flags.mark_flag_as_required('report_file')
    flags.mark_flag_as_required('eval_data')
    app.run(main)
