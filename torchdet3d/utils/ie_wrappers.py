import random
from copy import deepcopy as copy
from collections import namedtuple
import queue

import cv2 as cv
import glog as log
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, cdist

__all__ = ['Regressor', 'Detector', 'MultiCameraTracker']
THE_BIGGEST_DISTANCE = 10.
TrackedObj = namedtuple('TrackedObj', 'rect label')

COLOR_PALETTE = [[0, 113, 188],
                 [216, 82, 24],
                 [236, 176, 31],
                 [125, 46, 141],
                 [118, 171, 47],
                 [76, 189, 237],
                 [161, 19, 46],
                 [76, 76, 76],
                 [153, 153, 153],
                 [255, 0, 0],
                 [255, 127, 0],
                 [190, 190, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [170, 0, 255],
                 [84, 84, 0],
                 [84, 170, 0],
                 [84, 255, 0],
                 [170, 84, 0],
                 [170, 170, 0],
                 [170, 255, 0],
                 [255, 84, 0],
                 [255, 170, 0],
                 [255, 255, 0],
                 [0, 84, 127],
                 [0, 170, 127],
                 [0, 255, 127],
                 [84, 0, 127],
                 [84, 84, 127],
                 [84, 170, 127],
                 [84, 255, 127],
                 [170, 0, 127],
                 [170, 84, 127],
                 [170, 170, 127],
                 [170, 255, 127],
                 [255, 0, 127],
                 [255, 84, 127],
                 [255, 170, 127],
                 [255, 255, 127],
                 [0, 84, 255],
                 [0, 170, 255],
                 [0, 255, 255],
                 [84, 0, 255],
                 [84, 84, 255],
                 [84, 170, 255],
                 [84, 255, 255],
                 [170, 0, 255],
                 [170, 84, 255],
                 [170, 170, 255],
                 [170, 255, 255],
                 [255, 0, 255],
                 [255, 84, 255],
                 [255, 170, 255],
                 [42, 0, 0],
                 [84, 0, 0],
                 [127, 0, 0],
                 [170, 0, 0],
                 [212, 0, 0],
                 [255, 0, 0],
                 [0, 42, 0],
                 [0, 84, 0],
                 [0, 127, 0],
                 [0, 170, 0],
                 [0, 212, 0],
                 [0, 255, 0],
                 [0, 0, 42],
                 [0, 0, 84],
                 [0, 0, 127],
                 [0, 0, 170],
                 [0, 0, 212],
                 [0, 0, 255],
                 [0, 0, 0],
                 [36, 36, 36],
                 [72, 72, 72],
                 [109, 109, 109],
                 [145, 145, 145],
                 [182, 182, 182],
                 [218, 218, 218],
                 [255, 255, 255]]

class AverageEstimator(object):
    def __init__(self, initial_val=None):
        self.reset()
        if initial_val is not None:
            self.update(initial_val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def is_valid(self):
        return self.count > 0

    def merge(self, other):
        self.val = (self.val + other.val) * 0.5
        self.sum += other.sum
        self.count += other.count
        if self.count > 0:
            self.avg = self.sum / self.count

    def get(self):
        return self.avg

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
        return [res[key] for key in self.output_key]

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
    out_blob = net.outputs
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
            out = self.__decode_detections(out[0], rect)
            outputs.append(out)
        return outputs

    def __decode_detections(self, out, rect):
        """Decodes raw regression model output"""
        label = np.argmax(out[0])
        kp = out[1][label]
        kp = self.transform_kp(kp[0], rect[:4])

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



class Analyzer(object):
    def __init__(self, cam_id, enable,
                 show_distances=True,
                 concatenate_imgs_with_distances=True,
                 plot_timeline_freq=0,
                 save_distances='',
                 save_timeline='',
                 crop_size=(32, 64)):
        self.enable = enable
        self.id = cam_id
        self.show_distances = show_distances
        self.concatenate_distances = concatenate_imgs_with_distances
        self.plot_timeline_freq = plot_timeline_freq

        self.save_distances = os.path.join(save_distances, 'sct_{}'.format(cam_id)) \
            if len(save_distances) else ''
        self.save_timeline = os.path.join(save_timeline, 'sct_{}'.format(cam_id)) \
            if len(save_timeline) else ''

        if self.save_distances and not os.path.exists(self.save_distances):
            os.makedirs(self.save_distances)
        if self.save_timeline and not os.path.exists(self.save_timeline):
            os.makedirs(self.save_timeline)

        self.dist_names = ['Latest_feature', 'Average_feature', 'Cluster_feature', 'GIoU', 'Affinity_matrix']
        self.distance_imgs = [None for _ in range(len(self.dist_names))]
        self.current_detections = []  # list of numpy arrays
        self.crop_size = crop_size  # w x h

    def prepare_distances(self, tracks, current_detections):
        tracks_num = len(tracks)
        detections_num = len(current_detections)
        w, h = self.crop_size

        target_height = detections_num + 2
        target_width = tracks_num + 2

        img_size = (
            self.crop_size[1] * target_height,
            self.crop_size[0] * target_width, 3
        )

        for j, dist_img in enumerate(self.distance_imgs):
            self.distance_imgs[j] = np.full(img_size, 225, dtype='uint8')
            dist_img = self.distance_imgs[j]
            # Insert IDs:
            # 1. Tracked objects
            for i, track in enumerate(tracks):
                id = str(track.id)
                dist_img = cv.putText(dist_img, id, ((i + 2) * w + 5, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 2. Current detections
            for i, det in enumerate(current_detections):
                id = str(i)
                dist_img = cv.putText(dist_img, id, (5, (i + 2) * h + 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Insert crops
            # 1. Tracked objects (the latest crop)
            for i, track in enumerate(tracks):
                crop = track.crops[-1]
                y0, y1, x0, x1 = h, h * 2, (i + 2) * w, (i + 2) * w + w
                dist_img[y0: y1, x0: x1, :] = crop
            # 2. Current detections
            for i, det in enumerate(current_detections):
                dist_img[(i + 2) * h: (i + 2) * h + h, w: w * 2, :] = det
            # Insert grid line
            for n, i in enumerate(range(self.crop_size[1], dist_img.shape[0] + 1, self.crop_size[1])):
                x0, y0, x1, y1 = 0, i, dist_img.shape[1] - 1, i
                x0 = self.crop_size[0] * 2 if n < 1 else x0
                cv.line(dist_img, (x0, y0 - 1), (x1, y1 - 1), (0, 0, 0), 1, 1)
            for n, i in enumerate(range(0, dist_img.shape[1] + 1, self.crop_size[0])):
                x0, y0, x1, y1 = i, 0, i, dist_img.shape[0] - 1
                y0 = self.crop_size[1] * 2 if n == 1 else y0
                cv.line(dist_img, (x0 - 1, y0), (x1 - 1, y1), (0, 0, 0), 1, 1)
            # Insert hat
            x0, y0, x1, y1 = 0, 0, self.crop_size[0] * 2, self.crop_size[1] * 2
            cv.line(dist_img, (x0, y0), (x1, y1), (0, 0, 0), 1, 1)
            dist_img = cv.putText(dist_img, 'Tracks', (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            dist_img = cv.putText(dist_img, 'Detect', (4, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def visualize_distances(self, id_track=0, id_det=0, distances=None, affinity_matrix=None, active_tracks_idx=None):
        w, h = self.crop_size
        if affinity_matrix is None:
            for k, dist in enumerate(distances):
                value = str(dist)[:4] if dist else ' -'
                dist_img = self.distance_imgs[k]
                position = ((id_track + 2) * w + 1, (id_det + 2) * h + 24)
                dist_img = cv.putText(dist_img, value, position, cv.FONT_HERSHEY_SIMPLEX, 0.41, (0, 0, 0), 1)
        else:
            dist_img = self.distance_imgs[-1]
            for i in range(affinity_matrix.shape[0]):
                for j in range(affinity_matrix.shape[1]):
                    value = str(affinity_matrix[i][j])[:4] if affinity_matrix[i][j] else ' -'
                    track_id = active_tracks_idx[j]
                    position = ((track_id + 2) * w + 1, (i + 2) * h + 24)
                    dist_img = cv.putText(dist_img, value, position, cv.FONT_HERSHEY_SIMPLEX, 0.41, (0, 0, 0), 1)

    def show_all_dist_imgs(self, time, active_tracks):
        if self.distance_imgs[0] is None or not active_tracks:
            return
        concatenated_dist_img = None
        if self.concatenate_distances:
            for i, img in enumerate(self.distance_imgs):
                width = img.shape[1]
                height = 32
                title = np.full((height, width, 3), 225, dtype='uint8')
                title = cv.putText(title, self.dist_names[i], (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv.line(title, (0, height - 1), (width - 1, height - 1), (0, 0, 0), 1, 1)
                cv.line(title, (width - 1, 0), (width - 1, height - 1), (0, 0, 0), 1, 1)
                img = np.vstack([title, img])
                self.distance_imgs[i] = img
            concatenated_dist_img = np.hstack([self.distance_imgs[i] for i in range(0, 3)])
            concatenated_iou_am_img = np.hstack([self.distance_imgs[i] for i in range(3, 5)])
            empty_img = np.full(self.distance_imgs[2].shape, 225, dtype='uint8')
            concatenated_iou_am_img = np.hstack([concatenated_iou_am_img, empty_img])
            concatenated_dist_img = np.vstack([concatenated_dist_img, concatenated_iou_am_img])

        if self.show_distances:
            if concatenated_dist_img is not None:
                cv.imshow('SCT_{}_Distances'.format(self.id), concatenated_dist_img)
            else:
                for i, img in enumerate(self.distance_imgs):
                    cv.imshow(self.dist_names[i], img)
        if len(self.save_distances):
            if concatenated_dist_img is not None:
                file_path = os.path.join(self.save_distances, 'frame_{}_dist.jpg'.format(time))
                cv.imwrite(file_path, concatenated_dist_img)
            else:
                for i, img in enumerate(self.distance_imgs):
                    file_path = os.path.join(self.save_distances, 'frame_{}_{}.jpg'.format(time, self.dist_names[i]))
                    cv.imwrite(file_path, img)

    def plot_timeline(self, id, time, tracks):
        if self.plot_timeline_freq > 0 and time % self.plot_timeline_freq == 0:
            plot_timeline(id, time, tracks, self.save_timeline,
                          name='SCT', show_online=self.plot_timeline_freq)


def save_embeddings(scts, save_path, use_images=False, step=0):
    def make_label_img(label_img, crop, target_size=(32, 32)):
        img = cv.resize(crop, target_size)  # Resize, size must be square
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # BGR to RGB
        img = np.transpose(img, (2, 0, 1)) / 255  # Scale
        label_img = np.expand_dims(img, 0) if label_img is None else \
            np.concatenate((label_img, np.expand_dims(img, 0)))
        return label_img

    embeddings_all = None
    embeddings_avg = None
    embeddings_clust = None
    metadata_avg = []
    metadata_all = []
    metadata_clust = []
    label_img_all = None
    label_img_avg = None
    label_img_clust = None
    for i, sct in enumerate(scts):
        for track in tqdm(sct.tracks, 'Processing embeddings: SCT#{}...'.format(i)):
            if use_images and len(track.crops) == 1 and track.crops[0] is None:
                logging.warning('For embeddings was enabled parameter \'use_images\' but images were not found!'
                            '\'use_images\' switched off. Please check if parameter \'enable\' for analyzer'
                            'is set to True')
                use_images = False
            # Collect average embeddings
            if isinstance(track.f_avg.avg, int):
                continue
            embeddings_avg = track.f_avg.avg.reshape((1, -1)) if embeddings_avg is None else \
                np.concatenate((embeddings_avg, track.f_avg.avg.reshape((1, -1))))
            metadata_avg.append('sct_{}_'.format(i) + str(track.id))
            if use_images:
                label_img_avg = make_label_img(label_img_avg, track.crops[0])
            # Collect all embeddings
            features = None
            offset = 0
            for j, f in enumerate(track.features):
                if f is None:
                    offset += 1
                    continue
                features = f.reshape((1, -1)) if features is None else \
                    np.concatenate((features, f.reshape((1, -1))))
                metadata_all.append(track.id)
                if use_images:
                    crop = track.crops[j - offset]
                    label_img_all = make_label_img(label_img_all, crop)
            embeddings_all = features if embeddings_all is None else \
                np.concatenate((embeddings_all, features))
            # Collect clustered embeddings
            for j, f_clust in enumerate(track.f_clust.clusters):
                embeddings_clust = f_clust.reshape((1, -1)) if embeddings_clust is None else \
                                    np.concatenate((embeddings_clust, f_clust.reshape((1, -1))))
                metadata_clust.append(str(track.id))
                if use_images:
                    label_img_clust = make_label_img(label_img_clust, track.crops[j])

def plot_timeline(sct_id, last_frame_num, tracks, save_path='', name='', show_online=False):
    def find_max_id():
        max_id = 0
        for track in tracks:
            if isinstance(track, dict):
                track_id = track['id']
            else:
                track_id = track.id
            if track_id > max_id:
                max_id = track_id
        return max_id

    if not show_online and not len(save_path):
        return
    plot_name = '{}#{}'.format(name, sct_id)
    plt.figure(plot_name, figsize=(24, 13.5))
    last_id = find_max_id()
    xy = np.full((last_id + 1, last_frame_num + 1), -1, dtype='int32')
    x = np.arange(last_frame_num + 1, dtype='int32')
    y = np.arange(last_id + 1, dtype='int32')

    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Frame')
    plt.ylabel('Identity')

    colors = []
    for track in tracks:
        if isinstance(track, dict):
            frame_ids = track['timestamps']
            track_id = track['id']
        else:
            frame_ids = track.timestamps
            track_id = track.id
        if frame_ids[-1] > last_frame_num:
            frame_ids = [timestamp for timestamp in frame_ids if timestamp < last_frame_num]
        xy[track_id][frame_ids] = track_id
        xx = np.where(xy[track_id] == -1, np.nan, x)
        if track_id >= 0:
            color = COLOR_PALETTE[track_id % len(COLOR_PALETTE)] if track_id >= 0 else (0, 0, 0)
            color = [x / 255 for x in color]
        else:
            color = (0, 0, 0)
        colors.append(tuple(color[::-1]))
        plt.plot(xx, xy[track_id], marker=".", color=colors[-1], label='ID#{}'.format(track_id))
    if save_path:
        file_name = os.path.join(save_path, 'timeline_{}.jpg'.format(plot_name))
        plt.savefig(file_name, bbox_inches='tight')
    if show_online:
        plt.draw()
        plt.pause(0.01)

class ClusterFeature:
    def __init__(self, feature_len, initial_feature=None):
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        if initial_feature is not None:
            self.clusters.append(initial_feature)
            self.clusters_sizes.append(1)

    def update(self, feature_vec):
        if len(self.clusters) < self.feature_len:
            self.clusters.append(feature_vec)
            self.clusters_sizes.append(1)
        elif sum(self.clusters_sizes) < 2*self.feature_len:
            idx = random.randint(0, self.feature_len - 1)
            self.clusters_sizes[idx] += 1
            self.clusters[idx] += (feature_vec - self.clusters[idx]) / \
                                            self.clusters_sizes[idx]
        else:
            distances = cdist(feature_vec.reshape(1, -1),
                              np.array(self.clusters).reshape(len(self.clusters), -1), 'cosine')
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += 1
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / \
                                            self.clusters_sizes[nearest_idx]

    def merge(self, features, other, other_features):
        if len(features) > len(other_features):
            for feature in other_features:
                if feature is not None:
                    self.update(feature)
        else:
            for feature in features:
                if feature is not None:
                    other.update(feature)
            self.clusters = copy(other.clusters)
            self.clusters_sizes = copy(other.clusters_sizes)

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


class OrientationFeature:
    def __init__(self, feature_len, initial_feature=(None, None)):
        assert feature_len > 0
        self.orientation_features = [AverageEstimator() for _ in range(feature_len)]
        self.is_initialized = False
        if initial_feature[0] is not None and initial_feature[1] is not None and initial_feature[1] >= 0:
            self.is_initialized = True
            self.orientation_features[initial_feature[1]].update(initial_feature[0])

    def is_valid(self):
        return self.is_initialized

    def update(self, new_feature, idx):
        if idx >= 0:
            self.is_initialized = True
            self.orientation_features[idx].update(new_feature)

    def merge(self, other):
        for f1, f2 in zip(self.orientation_features, other.orientation_features):
            f1.merge(f2)
            self.is_initialized |= f1.is_valid()

    def dist_to_other(self, other):
        distances = [1.]
        for f1, f2 in zip(self.orientation_features, other.orientation_features):
            if f1.is_valid() and f2.is_valid():
                distances.append(0.5 * cosine(f1.get(), f2.get()))
        return min(distances)

    def dist_to_vec(self, vec, orientation):
        assert orientation < len(self.orientation_features)
        if orientation >= 0 and self.orientation_features[orientation].is_valid():
            return 0.5 * cosine(vec, self.orientation_features[orientation].get())
        return 1.


def clusters_distance(clusters1, clusters2):
    if len(clusters1) > 0 and len(clusters2) > 0:
        distances = 0.5 * cdist(clusters1.get_clusters_matrix(),
                                clusters2.get_clusters_matrix(), 'cosine')
        return np.amin(distances)
    return 1.


def clusters_vec_distance(clusters, feature):
    if len(clusters) > 0 and feature is not None:
        distances = 0.5 * cdist(clusters.get_clusters_matrix(),
                                feature.reshape(1, -1), 'cosine')
        return np.amin(distances)
    return 1.


class Track:
    def __init__(self, id, cam_id, box, time, feature=None, num_clusters=4, crops=None, orientation=None):
        self.id = id
        self.cam_id = cam_id
        self.f_avg = AverageEstimator()
        self.f_clust = ClusterFeature(num_clusters)
        self.f_orient = OrientationFeature(4, (feature, orientation))
        self.features = [feature]
        self.boxes = [box]
        self.timestamps = [time]
        self.crops = [crops]
        if feature is not None:
            self.f_avg.update(feature)
            self.f_clust.update(feature)

    def get_last_feature(self):
        return self.features[-1]

    def get_end_time(self):
        return self.timestamps[-1]

    def get_start_time(self):
        return self.timestamps[0]

    def get_last_box(self):
        return self.boxes[-1]

    def __len__(self):
        return len(self.timestamps)

    def _interpolate(self, target_box, timestamp, skip_size):
        last_box = self.get_last_box()
        for t in range(1, skip_size):
            interp_box = [int(b1 + (b2 - b1) / skip_size * t) for b1, b2 in zip(last_box, target_box)]
            self.boxes.append(interp_box)
            self.timestamps.append(self.get_end_time() + 1)
            self.features.append(None)

    def _filter_last_box(self, filter_speed):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            filtered_box = list(self.boxes[-2])
            for j in range(len(self.boxes[-1])):
                filtered_box[j] = int((1 - filter_speed) * filtered_box[j]
                                      + filter_speed * self.boxes[-1][j])
            self.boxes[-1] = tuple(filtered_box)

    def add_detection(self, box, feature, timestamp, max_skip_size=1, filter_speed=0.7, crop=None):
        skip_size = timestamp - self.get_end_time()
        if 1 < skip_size <= max_skip_size:
            self._interpolate(box, timestamp, skip_size)
            assert self.get_end_time() == timestamp - 1

        self.boxes.append(box)
        self.timestamps.append(timestamp)
        self.features.append(feature)
        self._filter_last_box(filter_speed)
        if feature is not None:
            self.f_clust.update(feature)
            self.f_avg.update(feature)
        if crop is not None:
            self.crops.append(crop)

    def merge_continuation(self, other, interpolate_time_thresh=0):
        assert self.get_end_time() < other.get_start_time()
        skip_size = other.get_start_time() - self.get_end_time()
        if 1 < skip_size <= interpolate_time_thresh:
            self._interpolate(other.boxes[0], other.get_start_time(), skip_size)
            assert self.get_end_time() == other.get_start_time() - 1

        self.f_avg.merge(other.f_avg)
        self.f_clust.merge(self.features, other.f_clust, other.features)
        self.f_orient.merge(other.f_orient)
        self.timestamps += other.timestamps
        self.boxes += other.boxes
        self.features += other.features
        self.crops += other.crops


class MultiCameraTracker:
    def __init__(self, num_sources, reid_model,
                 sct_config={},
                 time_window=20,
                 global_match_thresh=0.35,
                 bbox_min_aspect_ratio=1.2,
                 visual_analyze=None,
                 ):
        self.scts = []
        self.time = 0
        self.last_global_id = 0
        self.global_ids_queue = queue.Queue()
        assert time_window >= 1
        self.time_window = time_window  # should be greater than time window in scts
        assert 0 <= global_match_thresh <= 1
        self.global_match_thresh = global_match_thresh
        assert bbox_min_aspect_ratio >= 0
        self.bbox_min_aspect_ratio = bbox_min_aspect_ratio
        assert num_sources > 0
        for i in range(num_sources):
            self.scts.append(SingleCameraTracker(i, self._get_next_global_id,
                                                 self._release_global_id,
                                                 reid_model, visual_analyze=visual_analyze, **sct_config))

    def process(self, frames, all_detections, masks=None):
        assert len(frames) == len(all_detections) == len(self.scts)
        all_tracks = []
        for i, sct in enumerate(self.scts):
            if masks:
                mask = masks[i]
            else:
                mask = None
            if self.bbox_min_aspect_ratio is not None:
                all_detections[i], mask = self._filter_detections(all_detections[i], mask)
            sct.process(frames[i], all_detections[i], mask)
            all_tracks += sct.get_tracks()

        if self.time > 0 and self.time % self.time_window == 0:
            self._merge_all(all_tracks)

        self.time += 1

    def _merge_all(self, all_tracks):
        distance_matrix = self._compute_mct_distance_matrix(all_tracks)
        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.global_match_thresh:
                idx1, idx2 = indices_rows[i], indices_cols[j]
                if all_tracks[idx1].id > all_tracks[idx2].id:
                    self.scts[all_tracks[idx1].cam_id].check_and_merge(all_tracks[idx2], all_tracks[idx1])
                else:
                    self.scts[all_tracks[idx2].cam_id].check_and_merge(all_tracks[idx1], all_tracks[idx2])
                assert i != j
                distance_matrix = np.delete(distance_matrix, max(i, j), 0)
                distance_matrix = np.delete(distance_matrix, max(i, j), 1)
                distance_matrix = np.delete(distance_matrix, min(i, j), 0)
                distance_matrix = np.delete(distance_matrix, min(i, j), 1)
                indices_rows = np.delete(indices_rows, max(i, j))
                indices_rows = np.delete(indices_rows, min(i, j))
                indices_cols = np.delete(indices_cols, max(i, j))
                indices_cols = np.delete(indices_cols, min(i, j))
            else:
                break

    def _filter_detections(self, detections, masks):
        clean_detections = []
        clean_masks = []
        for i, det in enumerate(detections):
            w = det[2] - det[0]
            h = det[3] - det[1]
            ar = h / w
            if ar > self.bbox_min_aspect_ratio:
                clean_detections.append(det)
                if i < len(masks):
                    clean_masks.append(masks[i])
        return clean_detections, clean_masks

    def _compute_mct_distance_matrix(self, all_tracks):
        distance_matrix = THE_BIGGEST_DISTANCE * np.eye(len(all_tracks), dtype=np.float32)
        for i, track1 in enumerate(all_tracks):
            for j, track2 in enumerate(all_tracks):
                if j >= i:
                    break
                if track1.id != track2.id and track1.cam_id != track2.cam_id and \
                        len(track1) > self.time_window and len(track2) > self.time_window and \
                        track1.f_avg.is_valid() and track2.f_avg.is_valid():
                    if not track1.f_orient.is_valid():
                        f_complex_dist = clusters_distance(track1.f_clust, track2.f_clust)
                    else:
                        f_complex_dist = track1.f_orient.dist_to_other(track2.f_orient)
                    f_avg_dist = 0.5 * cosine(track1.f_avg.get(), track2.f_avg.get())
                    distance_matrix[i, j] = min(f_avg_dist, f_complex_dist)
                else:
                    distance_matrix[i, j] = THE_BIGGEST_DISTANCE
        return distance_matrix + np.transpose(distance_matrix)

    def _get_next_global_id(self):
        if self.global_ids_queue.empty():
            self.global_ids_queue.put(self.last_global_id)
            self.last_global_id += 1

        return self.global_ids_queue.get_nowait()

    def _release_global_id(self, id):
        assert id <= self.last_global_id
        self.global_ids_queue.put(id)

    def get_tracked_objects(self):
        return [sct.get_tracked_objects() for sct in self.scts]

    def get_all_tracks_history(self):
        history = []
        for sct in self.scts:
            cam_tracks = sct.get_archived_tracks() + sct.get_tracks()
            for i in range(len(cam_tracks)):
                cam_tracks[i] = {'id': cam_tracks[i].id,
                                 'timestamps': cam_tracks[i].timestamps,
                                 'boxes': cam_tracks[i].boxes}
            history.append(cam_tracks)
        return history

class SingleCameraTracker:
    def __init__(self, id, global_id_getter, global_id_releaser,
                 reid_model=None,
                 time_window=10,
                 continue_time_thresh=2,
                 track_clear_thresh=3000,
                 match_threshold=0.4,
                 merge_thresh=0.35,
                 n_clusters=4,
                 max_bbox_velocity=0.2,
                 detection_occlusion_thresh=0.7,
                 track_detection_iou_thresh=0.5,
                 process_curr_features_number=0,
                 visual_analyze=None,
                 interpolate_time_thresh=10,
                 detection_filter_speed=0.7,
                 rectify_thresh=0.25):
        self.reid_model = reid_model
        self.global_id_getter = global_id_getter
        self.global_id_releaser = global_id_releaser
        self.id = id
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
        assert 0 <= merge_thresh <= 1
        self.merge_thresh = merge_thresh
        assert n_clusters >= 1
        self.n_clusters = n_clusters
        assert 0 <= max_bbox_velocity
        self.max_bbox_velocity = max_bbox_velocity
        assert 0 <= detection_occlusion_thresh <= 1
        self.detection_occlusion_thresh = detection_occlusion_thresh
        assert 0 <= track_detection_iou_thresh <= 1
        self.track_detection_iou_thresh = track_detection_iou_thresh
        self.process_curr_features_number = process_curr_features_number
        assert interpolate_time_thresh >= 0
        self.interpolate_time_thresh = interpolate_time_thresh
        assert 0 <= detection_filter_speed <= 1
        self.detection_filter_speed = detection_filter_speed
        self.rectify_time_thresh = self.continue_time_thresh * 4
        self.rectify_length_thresh = self.time_window // 2
        assert 0 <= rectify_thresh <= 1
        self.rectify_thresh = rectify_thresh

        self.analyzer = None
        self.current_detections = None

        if visual_analyze is not None:
            self.analyzer = Analyzer(self.id, **visual_analyze)

    def process(self, frame, detections, mask=None):
        reid_features = [None]*len(detections)
        if self.reid_model:
            reid_features = self._get_embeddings(frame, detections, mask)

        assignment = self._continue_tracks(detections, reid_features)
        self._create_new_tracks(detections, reid_features, assignment)
        self._clear_old_tracks()
        self._rectify_tracks()
        if self.time % self.time_window == 0:
            self._merge_tracks()
        if self.analyzer:
            self.analyzer.plot_timeline(self.id, self.time, self.tracks)
        self.time += 1

    def get_tracked_objects(self):
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track.get_end_time() == self.time - 1 and len(track) > self.time_window:
                objs.append(TrackedObj(track.get_last_box(),
                                       label + ' ' + str(track.id)))
            elif track.get_end_time() == self.time - 1 and len(track) <= self.time_window:
                objs.append(TrackedObj(track.get_last_box(), label + ' -1'))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def check_and_merge(self, track_source, track_candidate):
        id_candidate = track_source.id
        idx = -1
        for i, track in enumerate(self.tracks):
            if track.boxes == track_candidate.boxes:
                idx = i
        if idx < 0:  # in this case track already has been modified, merge is invalid
            return

        collisions_found = False
        for i, hist_track in enumerate(self.history_tracks):
            if hist_track.id == id_candidate \
                and not (hist_track.get_end_time() < self.tracks[idx].get_start_time()
                         or self.tracks[idx].get_end_time() < hist_track.get_start_time()):
                collisions_found = True
                break

        for i, track in enumerate(self.tracks):
            if track is not None and track.id == id_candidate:
                collisions_found = True
                break

        if not collisions_found:
            self.tracks[idx].id = id_candidate
            self.tracks[idx].f_clust.merge(self.tracks[idx].features,
                                           track_source.f_clust, track_source.features)
            track_candidate.f_clust = copy(self.tracks[idx].f_clust)
        self.tracks = list(filter(None, self.tracks))

    def _continue_tracks(self, detections, features):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.continue_time_thresh:
                active_tracks_idx.append(i)

        occluded_det_idx = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i != j and self._ios(det1, det2) > self.detection_occlusion_thresh:
                    occluded_det_idx.append(i)
                    features[i] = None
                    break

        cost_matrix = self._compute_detections_assignment_cost(active_tracks_idx, detections, features)

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
                    crop = self.current_detections[i] if self.current_detections is not None else None
                    self.tracks[idx].add_detection(detections[i], features[i],
                                                   self.time, self.continue_time_thresh,
                                                   self.detection_filter_speed, crop)
        return assignment

    def _clear_old_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track.get_end_time() < self.time - self.track_clear_thresh:
                track.features = []
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track.get_end_time() < self.time - self.continue_time_thresh \
                    and len(track) < self.time_window:
                self.global_id_releaser(track.id)
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

    def _rectify_tracks(self):
        active_tracks_idx = []
        not_active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.rectify_time_thresh \
                    and len(track) >= self.rectify_length_thresh:
                active_tracks_idx.append(i)
            elif len(track) >= self.rectify_length_thresh:
                not_active_tracks_idx.append(i)

        distance_matrix = np.zeros((len(active_tracks_idx),
                                    len(not_active_tracks_idx)), dtype=np.float32)
        for i, idx1 in enumerate(active_tracks_idx):
            for j, idx2 in enumerate(not_active_tracks_idx):
                distance_matrix[i, j] = self._get_rectification_distance(self.tracks[idx1], self.tracks[idx2])

        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.rectify_thresh:
                self._concatenate_tracks(active_tracks_idx[indices_rows[i]],
                                         not_active_tracks_idx[indices_cols[j]])
                distance_matrix = np.delete(distance_matrix, i, 0)
                indices_rows = np.delete(indices_rows, i)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_cols = np.delete(indices_cols, j)
            else:
                break
        self.tracks = list(filter(None, self.tracks))

    def _get_rectification_distance(self, track1, track2):
        if (track1.get_start_time() > track2.get_end_time()
            or track2.get_start_time() > track1.get_end_time()) \
                and track1.f_avg.is_valid() and track2.f_avg.is_valid() \
                and self._check_tracks_velocity_constraint(track1, track2):
            return clusters_distance(track1.f_clust, track2.f_clust)
        return THE_BIGGEST_DISTANCE

    def _merge_tracks(self):
        distance_matrix = self._get_merge_distance_matrix()

        tracks_indices = np.arange(distance_matrix.shape[0])

        while len(tracks_indices) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                kept_idx = self._concatenate_tracks(tracks_indices[i], tracks_indices[j])
                deleted_idx = tracks_indices[i] if kept_idx == tracks_indices[j] else tracks_indices[j]
                assert self.tracks[deleted_idx] is None
                if deleted_idx == tracks_indices[i]:
                    idx_to_delete = i
                    idx_to_update = j
                else:
                    assert deleted_idx == tracks_indices[j]
                    idx_to_delete = j
                    idx_to_update = i
                updated_row = self._get_updated_merge_distance_matrix_row(kept_idx,
                                                                          deleted_idx,
                                                                          tracks_indices)
                distance_matrix[idx_to_update, :] = updated_row
                distance_matrix[:, idx_to_update] = updated_row
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 0)
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 1)
                tracks_indices = np.delete(tracks_indices, idx_to_delete)
            else:
                break

        self.tracks = list(filter(None, self.tracks))

    def _get_merge_distance(self, track1, track2):
        if (track1.get_start_time() > track2.get_end_time()
            or track2.get_start_time() > track1.get_end_time()) \
                and track1.f_avg.is_valid() and track2.f_avg.is_valid() \
                and self._check_tracks_velocity_constraint(track1, track2):
            f_avg_dist = 0.5 * cosine(track1.f_avg.get(), track2.f_avg.get())
            if track1.f_orient.is_valid():
                f_complex_dist = track1.f_orient.dist_to_other(track2.f_orient)
            else:
                f_complex_dist = clusters_distance(track1.f_clust, track2.f_clust)
            return min(f_avg_dist, f_complex_dist)

        return THE_BIGGEST_DISTANCE

    def _get_merge_distance_matrix(self):
        distance_matrix = THE_BIGGEST_DISTANCE*np.eye(len(self.tracks), dtype=np.float32)
        for i, track1 in enumerate(self.tracks):
            for j, track2 in enumerate(self.tracks):
                if i < j:
                    distance_matrix[i, j] = self._get_merge_distance(track1, track2)
        distance_matrix += np.transpose(distance_matrix)
        return distance_matrix

    def _get_updated_merge_distance_matrix_row(self, update_idx, ignore_idx, alive_indices):
        distance_matrix = THE_BIGGEST_DISTANCE*np.ones(len(alive_indices), dtype=np.float32)
        for i, idx in enumerate(alive_indices):
            if idx != update_idx and idx != ignore_idx:
                distance_matrix[i] = self._get_merge_distance(self.tracks[update_idx], self.tracks[idx])
        return distance_matrix

    def _concatenate_tracks(self, i, idx):
        if self.tracks[i].get_end_time() < self.tracks[idx].get_start_time():
            self.tracks[i].merge_continuation(self.tracks[idx], self.interpolate_time_thresh)
            self.tracks[idx] = None
            return i
        else:
            assert self.tracks[idx].get_end_time() < self.tracks[i].get_start_time()
            self.tracks[idx].merge_continuation(self.tracks[i], self.interpolate_time_thresh)
            self.tracks[i] = None
            return idx

    def _create_new_tracks(self, detections, features, assignment):
        assert len(detections) == len(features)
        for i, j in enumerate(assignment):
            if j is None:
                crop = self.current_detections[i] if self.analyzer else None
                self.tracks.append(Track(self.global_id_getter(), self.id,
                                         detections[i], self.time, features[i],
                                         self.n_clusters, crop, None))

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections, features):
        cost_matrix = np.zeros((len(detections), len(active_tracks_idx)), dtype=np.float32)
        if self.analyzer and len(self.tracks) > 0:
            self.analyzer.prepare_distances(self.tracks, self.current_detections)

        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx].get_last_box()
            for j, d in enumerate(detections):
                iou_dist = 0.5 * (1 - self._giou(d, track_box))
                reid_dist_curr, reid_dist_avg, reid_dist_clust = None, None, None
                if self.tracks[idx].f_avg.is_valid() and features[j] is not None \
                        and self.tracks[idx].get_last_feature() is not None:
                    reid_dist_avg = 0.5 * cosine(self.tracks[idx].f_avg.get(), features[j])
                    reid_dist_curr = 0.5 * cosine(self.tracks[idx].get_last_feature(), features[j])

                    if self.process_curr_features_number > 0:
                        num_features = len(self.tracks[idx])
                        step = -(-num_features // self.process_curr_features_number)
                        step = step if step > 0 else 1
                        start_index = 0 if self.process_curr_features_number > 1 else num_features - 1
                        for s in range(start_index, num_features - 1, step):
                            if self.tracks[idx].features[s] is not None:
                                reid_dist_curr = min(reid_dist_curr, 0.5 * cosine(self.tracks[idx].features[s], features[j]))

                    reid_dist_clust = clusters_vec_distance(self.tracks[idx].f_clust, features[j])
                    reid_dist = min(reid_dist_avg, reid_dist_curr, reid_dist_clust)
                else:
                    reid_dist = 0.5
                cost_matrix[j, i] = iou_dist * reid_dist
                if self.analyzer:
                    self.analyzer.visualize_distances(idx, j, [reid_dist_curr, reid_dist_avg, reid_dist_clust, 1 - iou_dist])
        if self.analyzer:
            self.analyzer.visualize_distances(affinity_matrix=1 - cost_matrix, active_tracks_idx=active_tracks_idx)
            self.analyzer.show_all_dist_imgs(self.time, len(self.tracks))
        return cost_matrix

    @staticmethod
    def _area(box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)

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

    def _ios(self, b1, b2, a1=None, a2=None):
        # intersection over self
        if a1 is None:
            a1 = self._area(b1)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])
        return intersection / a1 if a1 > 0 else 0

    def _get_embeddings(self, frame, detections, mask=None):
        rois = []
        embeddings = []

        if self.analyzer:
            self.current_detections = []

        for i in range(len(detections)):
            rect = detections[i]
            left, top, right, bottom = rect
            crop = frame[top:bottom, left:right]
            if mask and len(mask[i]) > 0:
                crop = cv.bitwise_and(crop, crop, mask=mask[i])
            if left != right and top != bottom:
                rois.append(crop)

            if self.analyzer:
                self.current_detections.append(cv.resize(crop, self.analyzer.crop_size))

        if rois:
            embeddings = self.reid_model.forward(rois)
            assert len(rois) == len(embeddings)

        return embeddings

    def _check_tracks_velocity_constraint(self, track1, track2):
        if track1.get_end_time() < track2.get_start_time():
            return self._check_velocity_constraint(track1.get_last_box(), track1.get_end_time(),
                                                   track2.boxes[0], track2.get_start_time())
        else:
            return self._check_velocity_constraint(track2.get_last_box(), track2.get_end_time(),
                                                   track1.boxes[0], track1.get_start_time())

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
