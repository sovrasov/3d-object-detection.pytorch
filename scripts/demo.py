import argparse

import cv2 as cv
import numpy as np
import glog as log
from openvino.inference_engine import IECore
from dataclasses import dataclass, asdict

from torchdet3d.utils import draw_kp, Regressor, Detector, OBJECTRON_CLASSES, IOUTracker

@dataclass
class IOUTrackerConfig:
    time_window : int = 10
    continue_time_thresh : int = 5
    track_clear_thresh : int = 3000
    match_threshold : float = 0.4
    track_detection_iou_thresh : float = 0.5
    interpolate_time_thresh : float = 10
    detection_filter_speed : float = 0.7
    keypoints_filter_speed : float = 0.3
    add_treshold : float = .1
    no_updated_frames_treshold : int = 5
    align_kp : bool = False


def draw_detections(frame, reg_detections, det_detections, ids, reg_only=True):
    """Draws detections and labels"""
    for det_out, reg_out, id_ in zip(det_detections, reg_detections, ids):
        left, top, right, bottom = det_out[:4]
        kp = np.reshape(reg_out[0], (9,2))
        label = reg_out[1]
        label = OBJECTRON_CLASSES[label]
        if not reg_only:
            if id_ != 'ID -1':
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
            else:
                cv.rectangle(frame, (left, top), (right, bottom), (100, 100, 100), thickness=2)
        if kp is not None and id_ != 'ID -1':
            frame = draw_kp(frame, kp, None, RGB=False, normalized=False)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, detector, regressor, sct_config, write_video=False, resolution = (1280, 720)):
    """Starts the 3D object detection demo"""
    fps = 20
    fourcc = cv.VideoWriter_fourcc(*'mpeg')
    if write_video:
        vout = cv.VideoWriter()
        vout.open('output_video_demo.mp4',fourcc,fps,resolution,True)
    win_name = '3D-object-detection'
    tracker =  IOUTracker(**sct_config)
    has_frame, prev_frame = capture.read()
    prev_frame = cv.resize(prev_frame, resolution)
    if not has_frame:
        return
    detector.run_async(prev_frame)
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return
        frame = cv.resize(frame, resolution)
        detections = detector.wait_and_grab()
        detector.run_async(frame)
        outputs = regressor.get_detections(prev_frame, detections)
        kps = [out[0].reshape(-1) for out in outputs]
        tracker.process(prev_frame, detections, kps)
        tracked_objects = tracker.get_tracked_objects()
        # get output from tracker
        rectangles = [rect.rect for rect in tracked_objects]
        kps = [rect.kp for rect in tracked_objects]
        ids = [rect.label for rect in tracked_objects]
        # since we work with normilized kps within the tracker, now we should transform them back
        decoded_kps = [regressor.transform_kp(np.array(kp).reshape(9,2), rect[:4]) for kp, rect in zip(kps, rectangles)]
        aligned_outputs = [(kp, out[1]) for kp, out in zip(decoded_kps, outputs)]

        vis = draw_detections(prev_frame, aligned_outputs, rectangles, ids, reg_only=False)
        cv.imshow(win_name, vis)
        if write_video:
            vout.write(vis)
        prev_frame, frame = frame, prev_frame

    capture.release()
    if write_video:
        vout.release()
    cv.destroyAllWindows()

def main():
    """Prepares data for the 3d object detection demo"""

    parser = argparse.ArgumentParser(description='3d object detection live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')
    parser.add_argument('--resolution', type=int, nargs='+', help='capture resolution')
    parser.add_argument('--od_model', type=str, required=True)
    parser.add_argument('--reg_model', type=str, required=True)
    parser.add_argument('--det_tresh', type=float, required=False, default=0.7)
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)
    parser.add_argument('--write_video', action='store_true',
                        help='whether to save a demo video or not')
    args = parser.parse_args()
    sct_config = IOUTrackerConfig()
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
    assert cap.isOpened()
    ie = IECore()
    object_detector = Detector(ie, args.od_model, args.det_tresh, args.device, args.cpu_extension)
    regressor = Regressor(ie, args.reg_model, args.device, args.cpu_extension)
    # running demo
    run(args, cap, object_detector, regressor, asdict(sct_config), args.write_video, tuple(args.resolution))

if __name__ == '__main__':
    main()
