import argparse
import json
import os
from os import path as osp

from tqdm import tqdm
import cv2 as cv
import numpy as np

from objectron_helpers import load_annotation_sequence, grab_frames

lists_root_path = osp.abspath(os.path.join(osp.dirname(__file__), '../3rdparty/Objectron/index'))


ALL_CLASSES = ['bike', 'book', 'bottle', 'cereal_box', 'camera', 'chair', 'cup', 'laptop', 'shoe']
OBJECTRON_NUM_KPS = 9


def load_video_info(data_root, subset, classes):
    videos_info = []
    avg_vid_len = 0
    for cl in classes:
        with open(osp.join(lists_root_path, f'{cl}_annotations_{subset}'), 'r') as f:
            for line in f:
                ann_path = osp.join(data_root, 'annotation' + osp.sep + line.strip() + '.pbdata')
                ann = load_annotation_sequence(ann_path) # object class can be incorrect in annoatation
                for item in ann:
                    item[1] = cl
                assert len(ann) > 0
                avg_vid_len += len(ann)
                vid_path = osp.join(data_root, 'videos' + osp.sep + line.strip() + osp.sep + 'video.MOV')
                videos_info.append((vid_path, ann))

    avg_vid_len /= len(videos_info)
    return videos_info, avg_vid_len


def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return None


def decode_keypoints(keypoints, keypoint_size_list, size):
    keypoints = np.split(keypoints, np.array(np.cumsum(keypoint_size_list)))
    keypoints = [points.reshape(-1, 3) for points in keypoints]
    unwrap_mat = np.asarray([size[0], size[1], 1.], np.float32)
    keypoints = [
        np.multiply(keypoint, unwrap_mat).astype(int)[:, :-1]
            for keypoint in keypoints
    ][:len(keypoint_size_list)]
    for i, kp in enumerate(keypoints):
        assert len(kp) == keypoint_size_list[i]
        assert len(kp) == OBJECTRON_NUM_KPS
    return keypoints


def get_bboxes_from_keypoints(keypoints, num_objects, size, clip_bboxes=False):
    w, h = size
    bboxes = []
    num_valid = 0

    for i in range(num_objects):
        min_x = np.min(keypoints[i][:,0])
        min_y = np.min(keypoints[i][:,1])
        max_x = np.max(keypoints[i][:,0])
        max_y = np.max(keypoints[i][:,1])
        if clip_bboxes:
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(w - 1, max_x), min(h - 1, max_y)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        if min_x < 0 or min_y < 0 or max_x >= w or max_y >= h or bbox[2]*bbox[3] == 0:
            bboxes.append(None)
        else:
            bboxes.append(bbox)
            num_valid += 1

    if num_valid > 0:
        return bboxes

    return None


def save_2_coco(output_root, subset_name, data_info, obj_classes, fps_divisor,
                res_divisor, dump_images=False, clip_classes=[], debug=False):
    json_name = f'objectron_{subset_name}.json'
    ann_folder = osp.join(output_root, 'annotations')
    img_folder = osp.join(output_root, 'images')
    if not osp.isdir(ann_folder):
        os.makedirs(ann_folder, exist_ok=True)
    if not osp.isdir(img_folder):
        os.makedirs(img_folder, exist_ok=True)

    img_id = 0
    ann_id = 0

    stat = {'Total frames' : 0, 'Avg box size': [0, 0], 'Total boxes': 0}
    ann_dict = {}
    categories = [{"id": i + 1, "name": cl} for i, cl in enumerate(obj_classes)]
    class_2_id = {cl : i + 1 for i, cl in enumerate(obj_classes)}
    images_info = []
    annotations = []

    for item in tqdm(data_info):
        vid_path, annotation = item
        # assert get_video_frames_number(vid_path) == len(annotation)
        req_frames = []
        for frame_idx in range(len(annotation)):
            if frame_idx % fps_divisor == 0:
                req_frames.append(frame_idx)
        frames = grab_frames(vid_path, req_frames)

        for frame_idx, frame_ann in enumerate(annotation):
            if frame_idx not in frames:
                continue
            if frames[frame_idx] is None:
                print('Warning: missing frame in ' + vid_path)
                continue
            #object_keypoints_2d, object_categories, keypoint_size_list, annotation_types

            h, w = frames[frame_idx].shape[0] // res_divisor, frames[frame_idx].shape[1] // res_divisor
            keypoints = decode_keypoints(frame_ann[0], frame_ann[2], (w, h))
            num_objects = len(frame_ann[2])
            bboxes = get_bboxes_from_keypoints(keypoints, num_objects, (w, h),
                                               clip_bboxes=frame_ann[1] in clip_classes)
            if bboxes is None:
                continue

            image_info = {}
            image_info['id'] = img_id
            img_id += 1
            image_info['height'], image_info['width'] = h, w
            vid_name_idx = vid_path.find('batch-')
            image_info['file_name'] = osp.join('images',
                    frame_ann[1] + '_' + vid_path[vid_name_idx : vid_path.rfind(osp.sep)].replace(osp.sep, '_') + \
                    '_' + str(frame_idx) + '.jpg')
            images_info.append(image_info)
            stat['Total frames'] += 1

            if debug:
                # visual debug
                frames[frame_idx] = cv.resize(frames[frame_idx], (w, h))
                for kp_pixel in keypoints[0]:
                    cv.circle(frames[frame_idx], (kp_pixel[0], kp_pixel[1]), 5, (255, 0, 0), -1)
                if len(keypoints) > 1:
                    for kp_pixel in keypoints[1]:
                        cv.circle(frames[frame_idx], (kp_pixel[0], kp_pixel[1]), 5, (0, 0, 255), -1)
                for bbox in bboxes:
                    if bbox is not None:
                        cv.rectangle(frames[frame_idx], (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
                cv.imwrite(osp.join(output_root, image_info['file_name']), frames[frame_idx])

            if dump_images and not debug:
                frames[frame_idx] = cv.resize(frames[frame_idx], (w, h))
                cv.imwrite(osp.join(output_root, image_info['file_name']), frames[frame_idx])

            for i in range(num_objects):
                if bboxes[i] is not None:
                    stat['Total boxes'] += 1
                    stat['Avg box size'][0] += bboxes[i][2]
                    stat['Avg box size'][1] += bboxes[i][3]
                    ann = {
                        'id': ann_id,
                        'image_id': image_info['id'],
                        'segmentation': [],
                        'num_keypoints': frame_ann[2][i],
                        'keypoints': list(keypoints[i].reshape(-1)),
                        'category_id': class_2_id[frame_ann[1]],
                        'iscrowd': 0,
                        'area': bboxes[i][2] * bboxes[i][3],
                        'bbox': bboxes[i]
                        }
                    ann_id += 1
                    annotations.append(ann)

    ann_dict['images'] = images_info
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    with open(osp.join(ann_folder, json_name), 'w') as f:
        f.write(json.dumps(ann_dict, default=np_encoder))
    stat['Avg box size'][0] /= stat['Total boxes']
    stat['Avg box size'][1] /= stat['Total boxes']
    return stat


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, default='',
                        help='path to objectron raw data root', required=True)
    parser.add_argument('--output_folder', type=str, default='',
                        help='path to output folder with COCO annotation', required=True)
    parser.add_argument('--obj_classes', type=str, nargs='+', default='cereal_box', help='Classes to convert')
    parser.add_argument('--fps_divisor', type=int, default=1, help='')
    parser.add_argument('--res_divisor', type=int, default=1, help='')
    parser.add_argument('--only_annotation', action='store_true')
    args = parser.parse_args()

    if args.obj_classes[0] == 'all':
        args.obj_classes = ALL_CLASSES

    for cl in args.obj_classes:
        assert cl in ALL_CLASSES

    subsets = ['train', 'test']
    data_info = {}
    for subset in subsets:
        print(f'Loading {subset} metadata...')
        videos_info, avg_len = load_video_info(args.data_root, subset, args.obj_classes)
        print(f'# of {subset} videos: {len(videos_info)}, avg length: {avg_len}')
        data_info[subset] = videos_info

    for k in data_info:
        print('Converting ' + k)
        stat = save_2_coco(args.output_folder, k, data_info[k], args.obj_classes,
                           args.fps_divisor, args.res_divisor, not args.only_annotation, ['shoe', 'bike'])
        for c in stat:
            print(f'{c}: {stat[c]}')


if __name__ == '__main__':
    main()
