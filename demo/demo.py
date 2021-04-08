import argparse

import cv2 as cv
import glog as log
import numpy as np
from openvino.inference_engine import IECore

from demo_tools import load_ie_model
from torchdet3d.utils import draw_kp


OBJECTRON_CLASSES = ('bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe')

class Detector:
    """Wrapper class for object detector"""
    def __init__(self, ie,  model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1., 1.)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        out = self.net.forward(frame)
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


class Regressor:
    """Wrapper class for regression model"""
    def __init__(self, ie,  model_path, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)

    def get_detections(self, frame, detections):
        """Returns all detections on frame"""
        outputs = []
        for rect in detections:
            cropped_img = self.crop(frame, rect[0])
            out = self.net.forward(cropped_img)
            out = self.__decode_detections(out, rect)
            outputs.append(out)
        return outputs

    def __decode_detections(self, out, rect):
        """Decodes raw regression model output"""
        label = int(rect[2])
        kp = out[label]
        kp = self.transform_kp(kp[0], rect[0])

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

def draw_detections(frame, reg_detections, det_detections, reg_only=True):
    """Draws detections and labels"""
    for det_out, reg_out in zip(det_detections, reg_detections):
        left, top, right, bottom = det_out[0]
        kp = reg_out[0]
        label = reg_out[1]
        label = OBJECTRON_CLASSES[label]
        if not reg_only:
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)

        frame = draw_kp(frame, kp, None, RGB=False, normalized=False)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, detector, regressor, write_video=False, resolution = (1280, 720)):
    """Starts the 3D object detection demo"""
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    fps = 24
    if write_video:
        writer_video = cv.VideoWriter('output_video_demo.mp4', fourcc, fps, resolution)
    win_name = '3D-object-detection'
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        frame = cv.resize(frame, resolution)
        if not has_frame:
            return
        detections = detector.get_detections(frame)
        outputs = regressor.get_detections(frame, detections)

        frame = draw_detections(frame, outputs, detections, reg_only=False)
        cv.imshow(win_name, frame)
        if write_video:
            writer_video.write(cv.resize(frame, resolution))
            writer_video.release()
    capture.release()
    cv.destroyAllWindows()

def main():
    """Prepares data for the 3d object detection demo"""

    parser = argparse.ArgumentParser(description='3d object detection live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')
    parser.add_argument('--resolution', type=int, nargs='+', help='capture resolution')
    parser.add_argument('--config', type=str, default=None, required=False,
                        help='Configuration file')
    parser.add_argument('--od_model', type=str, required=True)
    parser.add_argument('--reg_model', type=str, required=True)
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
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.resolution[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.resolution[1])
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    else:
        assert args.video, "No video input was given"
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    assert cap.isOpened()
    ie = IECore()
    object_detector = Detector(ie, args.od_model, args.det_tresh, args.device, args.cpu_extension)
    regressor = Regressor(ie, args.reg_model, args.device, args.cpu_extension)
    # running demo
    run(args, cap, object_detector, regressor, args.write_video, tuple(args.resolution))

if __name__ == '__main__':
    main()
