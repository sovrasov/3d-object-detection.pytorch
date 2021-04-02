import argparse
import inspect
import os.path as osp
import os
import sys

import cv2 as cv
import glog as log
import numpy as np
from openvino.inference_engine import IECore

from demo_tools import load_ie_model


OBJECTRON_CLASSES = ('bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe')

class Detector:
    """Wrapper class for face detector"""
    def __init__(self, ie,  model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1., 1.)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        _, _, h, w = self.net.get_input_shape()
        out = self.net.forward(cv.resize(frame, (w, h)))
        detections = self.__decode_detections(out, frame.shape)
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

                detections.append(((left, top, right, bottom), confidence, label))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections

def draw_detections(frame, detections):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        conf = rect[1]
        label = OBJECTRON_CLASSES[int(rect[2])]
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, detector, write_video=False):
    """Starts the 3D object detection demo"""
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    resolution = (1280, 720)
    fps = 24
    if write_video:
        writer_video = cv.VideoWriter('output_video_demo.mp4', fourcc, fps, resolution)
    win_name = '3D-object-detection'
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return
        detections = detector.get_detections(frame)
        frame = draw_detections(frame, detections)
        cv.imshow(win_name, frame)
        if write_video:
            writer_video.write(cv.resize(frame, resolution))
            writer_video.release()
    capture.release()
    cv.destroyAllWindows()

def main():
    """Prepares data for the antispoofing recognition demo"""

    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')
    parser.add_argument('--config', type=str, default=None, required=False,
                        help='Configuration file')
    parser.add_argument('--fd_model', type=str, required=True)
    parser.add_argument('--det_tresh', type=float, required=False, default=0.6)
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)
    parser.add_argument('--write_video', type=bool, default=False,
                        help='if you set this arg to True, the video of the demo will be recoreded')
    args = parser.parse_args()

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
        cap = cv.VideoCapture(args.cam_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    else:
        assert args.video, "No video input was given"
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    assert cap.isOpened()
    ie = IECore()
    object_detector = Detector(ie, args.fd_model, args.det_tresh, args.device, args.cpu_extension)
    # running demo
    run(args, cap, object_detector, args.write_video)

if __name__ == '__main__':
    main()
