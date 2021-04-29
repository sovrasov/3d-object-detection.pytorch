import os
import sys
import subprocess

import cv2
import numpy as np

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

#pylint: disable = wrong-import-position
from objectron.schema import annotation_data_pb2 as annotation_protocol

# The annotations are stored in protocol buffer format.
# The AR Metadata captured with each frame in the video


def get_frame_annotation(sequence, frame_id):
    """Grab an annotated frame from the sequence."""
    data = sequence.frame_annotations[frame_id]
    object_id = 0
    object_keypoints_2d = []
    object_keypoints_3d = []
    object_rotations = []
    object_translations = []
    object_scale = []
    num_keypoints_per_object = []
    object_categories = []
    annotation_types = []
    # Get the camera for the current frame. We will use the camera to bring
    # the object from the world coordinate to the current camera coordinate.
    camera = np.array(data.camera.transform).reshape(4, 4)

    for obj in sequence.objects:
        rotation = np.array(obj.rotation).reshape(3, 3)
        translation = np.array(obj.translation)
        object_scale.append(np.array(obj.scale))
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        obj_cam = np.matmul(camera, transformation)
        object_translations.append(obj_cam[:3, 3])
        object_rotations.append(obj_cam[:3, :3])
        object_categories.append(obj.category)
        annotation_types.append(obj.type)

    keypoint_size_list = []
    for annotations in data.annotations:
        num_keypoints = len(annotations.keypoints)
        keypoint_size_list.append(num_keypoints)
        for keypoint_id in range(num_keypoints):
            keypoint = annotations.keypoints[keypoint_id]
            object_keypoints_2d.append(
                (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
            object_keypoints_3d.append(
                (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
        num_keypoints_per_object.append(num_keypoints)
        object_id += 1
    return [object_keypoints_2d, object_categories, keypoint_size_list,
            annotation_types]


def get_video_frames_number(video_file):
    capture = cv2.VideoCapture(video_file)
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def grab_frames(video_file, frame_ids, use_opencv=True):
    """Grab an image frame from the video file."""
    frames = {}
    capture = cv2.VideoCapture(video_file)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_opencv:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            _, current_frame = capture.read()
            frames[frame_id] = current_frame
        capture.release()
    else:
        frame_size = width * height * 3

        for frame_id in frame_ids:
            frame_filter = r'select=\'eq(n\,{:d})\''.format(frame_id)
            command = [
                'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', frame_filter,
                '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-'
            ]
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, bufsize=151 * frame_size, stderr=subprocess.DEVNULL) as pipe:
                current_frame = np.frombuffer(
                    pipe.stdout.read(frame_size), dtype='uint8').reshape(height, width, 3)
                pipe.stdout.flush()
                frames[frame_id] = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    return frames


def load_annotation_sequence(annotation_file):
    frame_annotations = []
    with open(annotation_file, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())
        for i in range(len(sequence.frame_annotations)):
            frame_annotations.append(get_frame_annotation(sequence, i))
           # annotation, cat, num_keypoints, types
    return frame_annotations
