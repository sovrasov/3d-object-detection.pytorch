from collections import namedtuple
import queue

import cv2 as cv
import glog as log
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


__all__ = ['Regressor', 'Detector', 'CameraTracker']
TrackedObj = namedtuple('TrackedObj', 'rect kp label')

class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs_info, input_key, output_key):
        self.net = exec_net
        self.inputs_info = inputs_info
        self.input_key = input_key
        self.output_key = output_key
        self.reqs_ids = []

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
        return list(res.values())

    def forward_async(self, img):
        id_ = len(self.reqs_ids)
        self.net.start_async(request_id=id_,
                             inputs={self.input_key: self._preprocess(img)})
        self.reqs_ids.append(id_)

    def grab_all_async(self):
        outputs = []
        for id_ in self.reqs_ids:
            self.net.requests[id_].wait(-1)
            output_list = [self.net.requests[id_].output_blobs[key].buffer for key in self.output_key]
            outputs.append(output_list)
        self.reqs_ids = []
        return outputs

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key].input_data.shape


def load_ie_model(ie, model_xml, device, plugin_dir, cpu_extension='', num_reqs=1):
    """Loads a model in the Inference Engine format"""
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing Inference Engine plugin for %s ", device)

    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, 'CPU')
    # Read IR
    log.info("Loading network")
    net = ie.read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = [key for key in net.outputs]
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device, num_requests=num_reqs)
    model = IEModel(exec_net, net.input_info, input_blob, out_blob)
    return model


class Detector:
    """Wrapper class for object detector"""
    def __init__(self, ie,  model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1., 1.)

    def run_async(self, frame):
        self.frame_shape = frame.shape
        self.net.forward_async(frame)

    def wait_and_grab(self):
        outputs = self.net.grab_all_async()
        detections = []
        assert len(outputs) == 1
        detections = self.__decode_detections(outputs[0][0], self.frame_shape)
        return detections

    def get_detections(self, frame):
        """Returns all detections on frame"""
        out = self.net.forward(frame)
        detections = self.__decode_detections(out[0], frame.shape)
        return detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            label = detection[1]
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append((left, top, right, bottom, confidence, label))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections


class Regressor:
    """Wrapper class for regression model"""
    def __init__(self, ie,  model_path, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)

    def get_detections(self, frame, detections):
        """Returns all detections on frame"""
        outputs = []
        for rect in detections:
            cropped_img = self.crop(frame, rect[:4])
            out = self.net.forward(cropped_img)
            out = self.__decode_detections(out, rect)
            outputs.append(out)
        return outputs

    def __decode_detections(self, out, rect):
        """Decodes raw regression model output"""
        label = np.argmax(out[0])
        kp = out[1][label]
        return (kp, label)

    @staticmethod
    def transform_kp(kp: np.array, crop_cords: tuple):
        x0,y0,x1,y1 = crop_cords
        crop_shape = (x1-x0,y1-y0)
        kp[:,0] = kp[:,0]*crop_shape[0]
        kp[:,1] = kp[:,1]*crop_shape[1]
        kp[:,0] += x0
        kp[:,1] += y0
        return kp

    @staticmethod
    def crop(frame, rect):
        x0, y0, x1, y1 = rect
        crop = frame[y0:y1, x0:x1]
        return crop


class Track:
    def __init__(self, ID, bbox, kps, time, align_kp=False):
        self.id = ID
        self.boxes = [bbox]
        self.kps = [kps]
        self.timestamps = [time]
        self.no_updated_frames = 0
        self.align_kp = align_kp

    def get_end_time(self):
        return self.timestamps[-1]

    def get_start_time(self):
        return self.timestamps[0]

    def get_last_box(self):
        return self.boxes[-1]

    def get_last_kp(self):
        return self.kps[-1]

    def __len__(self):
        return len(self.timestamps)

    def _interpolate(self, target_box, target_kp, timestamp, skip_size):
        last_box = self.get_last_box()
        last_kp = self.get_last_kp()
        for t in range(1, skip_size):
            interp_box = [int(b1 + (b2 - b1) / skip_size * t) for b1, b2 in zip(last_box, target_box)]
            interp_kp = [k1 + (k2 - k1) / skip_size * t for k1, k2 in zip(last_kp, target_kp)]
            self.boxes.append(interp_box)
            self.kps.append(interp_kp)
            self.timestamps.append(self.get_end_time() + 1)

    def _filter_last_3d_box(self, filter_speed, add_treshold, no_updated_frames_treshold):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            num_keypoints = len(self.kps[-2]) // 2
            self.kps[-2] = np.array(self.kps[-2]).reshape(num_keypoints,2)
            self.kps[-1] = self.kps[-1].reshape(num_keypoints,2)
            # compute average distance before run
            add_dist = np.mean(np.linalg.norm(self.kps[-1] - self.kps[-2], axis=1))
            if self.align_kp:
                indexes_to_revert = self._align_kp_positions()
                self.kps[-1] = self.kps[-1][indexes_to_revert]
            considered_kps = self.kps[-1]
            # if add distance is appropriate for previous frame by given treshold
            # then we smooth kps with EMA
            if add_dist < add_treshold:
                self.no_updated_frames = 0
                filtered_kps = (1 - filter_speed) * self.kps[-2] + filter_speed * considered_kps
            elif self.no_updated_frames > no_updated_frames_treshold:
                # if bbox haven't been updated too long -> interrupt EMA
                # and get new bbox
                filtered_kps = considered_kps
            else:
                # if not -> use bbox from previous frame
                filtered_kps = self.kps[-2]
                self.no_updated_frames += 1

            self.kps[-1] = tuple(filtered_kps.reshape(-1).tolist())

    def _align_kp_positions(self):
        # store indexes for matching
        indexes = dict(zip(range(9), range(9)))
        # list for marking vertexes
        ind_updated = [False for i in range(9)]
        for i in range(len(self.kps[-1])):
            if ind_updated[i]:
                continue
            distance = np.linalg.norm(self.kps[-1][i, :] - self.kps[-2][i, :])
            min_d_idx = i
            for j in range(i+1, len(self.kps[-1])):
                d = np.linalg.norm(self.kps[-1][i, :] - self.kps[-2][j, :])
                if d < distance:
                    min_d_idx = j
            # if we already rearranged vertexes we will not do it twice to prevent
            # indexes mess
            if min_d_idx != i and not ind_updated[i] and not ind_updated[min_d_idx]:
                # swap vertexes
                indexes[i] = min_d_idx
                indexes[min_d_idx] = i
                # mark vertexes as visited
                ind_updated[i] = True
                ind_updated[min_d_idx] = True

        return indexes

    def _filter_last_box(self, filter_speed):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            filtered_box = list(self.boxes[-2])
            for j in range(len(self.boxes[-1])):
                filtered_box[j] = int((1 - filter_speed) * filtered_box[j]
                                      + filter_speed * self.boxes[-1][j])
            self.boxes[-1] = tuple(filtered_box)

    def add_detection(self, bbox, kps, timestamp, max_skip_size=1,
                      box_filter_speed=0.7, kp_filter_speed=0.3,
                      add_treshold=0.1, no_updated_frames_treshold=5):
        skip_size = timestamp - self.get_end_time()
        if 1 < skip_size <= max_skip_size:
            self._interpolate(bbox, kps, timestamp, skip_size)
            assert self.get_end_time() == timestamp - 1

        self.boxes.append(bbox)
        self.kps.append(kps)
        self.timestamps.append(timestamp)
        self._filter_last_box(box_filter_speed)
        self._filter_last_3d_box(kp_filter_speed, add_treshold, no_updated_frames_treshold)


class CameraTracker:
    def __init__(self, sct_params):
        self.time = 0
        self.last_global_id = 0
        self.global_ids_queue = queue.Queue()
        self.sct = SingleCameraTracker(self._get_next_global_id,
                                            self._release_global_id, **sct_params)

    def process(self, frames, all_detections, all_kps):
        self.sct.process(frames, all_detections, all_kps)
        self.time += 1

    def _get_next_global_id(self):
        if self.global_ids_queue.empty():
            self.global_ids_queue.put(self.last_global_id)
            self.last_global_id += 1

        return self.global_ids_queue.get_nowait()

    def _release_global_id(self, ID):
        assert ID <= self.last_global_id
        self.global_ids_queue.put(ID)

    def get_tracked_objects(self):
        return self.sct.get_tracked_objects()


class SingleCameraTracker:
    def __init__(self, global_id_getter, global_id_releaser,
                 time_window=5,
                 continue_time_thresh=2,
                 track_clear_thresh=3000,
                 match_threshold=0.4,
                 max_bbox_velocity=0.2,
                 track_detection_iou_thresh=0.5,
                 interpolate_time_thresh=10,
                 detection_filter_speed=0.7,
                 keypoints_filter_speed=0.3,
                 add_treshold=0.1,
                 no_updated_frames_treshold=5):

        self.global_id_getter = global_id_getter
        self.global_id_releaser = global_id_releaser
        self.tracks = []
        self.history_tracks = []
        self.time = 0
        assert time_window >= 1
        self.time_window = time_window
        assert continue_time_thresh >= 1
        self.continue_time_thresh = continue_time_thresh
        assert track_clear_thresh >= 1
        self.track_clear_thresh = track_clear_thresh
        assert 0 <= match_threshold <= 1
        self.match_threshold = match_threshold
        assert max_bbox_velocity >= 0
        self.max_bbox_velocity = max_bbox_velocity
        assert 0 <= track_detection_iou_thresh <= 1
        self.track_detection_iou_thresh = track_detection_iou_thresh
        assert interpolate_time_thresh >= 0
        self.interpolate_time_thresh = interpolate_time_thresh
        assert 0 <= detection_filter_speed <= 1
        self.detection_filter_speed = detection_filter_speed
        assert 0 <= keypoints_filter_speed <= 1
        self.keypoints_filter_speed = keypoints_filter_speed
        assert 0 <= add_treshold <= 1
        self.add_treshold = add_treshold
        assert no_updated_frames_treshold >= 0
        assert isinstance(no_updated_frames_treshold, int)
        self.no_updated_frames_treshold = no_updated_frames_treshold
        self.current_detections = None

    def process(self, frame, detections, kps):
        assignment = self._continue_tracks(detections, kps)
        self._create_new_tracks(detections, kps, assignment)
        self._clear_old_tracks()
        self.time += 1

    def get_tracked_objects(self):
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track.get_end_time() == self.time - 1 and len(track) > self.time_window:
                objs.append(TrackedObj(track.get_last_box(), track.get_last_kp(),
                                       label + ' ' + str(track.id)))
            elif track.get_end_time() == self.time - 1 and len(track) <= self.time_window:
                objs.append(TrackedObj(track.get_last_box(), track.get_last_kp(), label + ' -1'))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def _continue_tracks(self, detections, kps):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.continue_time_thresh:
                active_tracks_idx.append(i)

        cost_matrix = self._compute_detections_assignment_cost(active_tracks_idx, detections)

        assignment = [None for _ in range(cost_matrix.shape[0])]
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                idx = active_tracks_idx[j]
                if cost_matrix[i, j] < self.match_threshold and \
                    self._check_velocity_constraint(self.tracks[idx].get_last_box(),
                                                    self.tracks[idx].get_end_time(),
                                                    detections[i], self.time) and \
                        self._iou(self.tracks[idx].boxes[-1], detections[i]) > self.track_detection_iou_thresh:
                    assignment[i] = j

            for i, j in enumerate(assignment):
                if j is not None:
                    idx = active_tracks_idx[j]
                    self.tracks[idx].add_detection(detections[i], kps[i],
                                                   self.time, self.continue_time_thresh,
                                                   self.detection_filter_speed, self.keypoints_filter_speed,
                                                   self.add_treshold, self.no_updated_frames_treshold)
        return assignment

    def _clear_old_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track.get_end_time() < self.time - self.track_clear_thresh:
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track.get_end_time() < self.time - self.continue_time_thresh \
                    and len(track) < self.time_window:
                self.global_id_releaser(track.id)
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections):
        cost_matrix = np.zeros((len(detections), len(active_tracks_idx)), dtype=np.float32)

        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx].get_last_box()
            for j, d in enumerate(detections):
                iou_dist = 0.5 * (1 - self._giou(d, track_box))
                cost_matrix[j, i] = iou_dist

        return cost_matrix

    def _create_new_tracks(self, detections, kps, assignment):
        for i, j in enumerate(assignment):
            if j is None:
                self.tracks.append(Track(self.global_id_getter(),
                                         detections[i], kps[i], self.time))

    @staticmethod
    def _area(bbox):
        return max((bbox[2] - bbox[0]), 0) * max((bbox[3] - bbox[1]), 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        enclosing = self._area([min(b1[0], b2[0]), min(b1[1], b2[1]),
                                max(b1[2], b2[2]), max(b1[3], b2[3])])
        u = a1 + a2 - intersection
        iou = intersection / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        u = a1 + a2 - intersection
        return intersection / u if u > 0 else 0

    def _check_velocity_constraint(self, detection1, det1_time, detection2, det2_time):
        dt = abs(det2_time - det1_time)
        avg_size = 0
        for det in [detection1, detection2]:
            avg_size += 0.5 * (abs(det[2] - det[0]) + abs(det[3] - det[1]))
        avg_size *= 0.5
        shifts = [abs(x - y) for x, y in zip(detection1, detection2)]
        velocity = sum(shifts) / len(shifts) / dt / avg_size
        if velocity > self.max_bbox_velocity:
            return False
        return True
