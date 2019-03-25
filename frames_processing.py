import cv2
import os
from collections import defaultdict
import xml.etree.ElementTree as ET

from bounding_box import BoundingBox
from utils import get_patch, scale_gt

train_target_folder = 'train\\target'
train_search_folder = 'train\\search'

data_folder = 'imagedata++'
ann_folder = 'alov_ann'

path_collector = defaultdict(dict)
current_dir_path = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.join(current_dir_path, train_target_folder)
search_path = os.path.join(current_dir_path, train_search_folder)

data_path = os.path.join(current_dir_path, data_folder)
ann_path = os.path.join(current_dir_path, ann_folder)

video_types = [type for type in os.listdir(data_path)]

for video_type in video_types:
    video_type_path = os.path.join(data_path, video_type)
    ann_video_type_path = os.path.join(ann_path, video_type)
    video_dict = defaultdict(dict)

    for video in os.listdir(video_type_path):
        video_path = os.path.join(video_type_path, video)
        video_attributes = defaultdict(list)

        for frame_name in os.listdir(video_path):
            video_attributes['frames'] += [os.path.join(video_path, frame_name)]

        ann_video_path = os.path.join(ann_video_type_path, video + '.xml')
        video_attributes['ann'] = ann_video_path

        video_dict[video] = video_attributes
    path_collector[video_type] = video_dict

patch_counter = 0
for video_type in path_collector.keys():
    for video in path_collector[video_type]:
        ann_file = path_collector[video_type][video]['ann']
        frames_path = path_collector[video_type][video]['frames']

        document = ET.parse(ann_file)
        root = document.getroot()
        end_frame = root[1].text

        for index, frame in enumerate(root):
            if index == len(root) - 1:
                break
            if frame.tag != 'frame':
                continue

            target_gt = BoundingBox()
            search_gt = BoundingBox()

            x_target_coords = [
                attr.text
                for item in frame if item.tag != 'number'
                for attr in item if attr.tag == 'x'
            ]
            y_target_coords = [
                attr.text
                for item in frame if item.tag != 'number'
                for attr in item if attr.tag == 'y'
            ]

            target_frame_number = root[index].find('number').text
            target_frame = cv2.imread(frames_path[int(target_frame_number) - 1])

            x_search_coords = [
                attr.text
                for item in root[index + 1] if item.tag != 'number'
                for attr in item if attr.tag == 'x'
            ]
            y_search_coords = [
                attr.text
                for item in root[index + 1] if item.tag != 'number'
                for attr in item if attr.tag == 'y'
            ]

            search_frame_number = root[index + 1].find('number').text
            search_frame = cv2.imread(frames_path[int(search_frame_number) - 1])

            x1_t = float(min(x_target_coords))
            y1_t = float(min(y_target_coords))
            x2_t = float(max(x_target_coords))
            y2_t = float(max(y_target_coords))

            x2_s = float(min(x_search_coords))
            y1_s = float(min(y_search_coords))
            x1_s = float(max(x_search_coords))
            y2_s = float(max(y_search_coords))

            target_gt .update_coordinates(x1_t, x2_t, y1_t, y2_t)
            search_gt.update_coordinates(x1_s, x2_s, y1_s, y2_s)

            target, target_coordinates = get_patch(target_frame, target_gt, 'target')
            search = get_patch(search_frame, target_gt, 'search', target_coordinates)

            cv2.imwrite(os.path.join(target_path, str(patch_counter) + '.jpg'), target)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(search_path, str(patch_counter) + '.jpg'), search)
            cv2.waitKey(0)

            scaled_gt = scale_gt(target_gt, search_gt, target)

            with open('train_file.txt', 'a') as train_file:
                t_patch_path = os.path.join(target_path, str(patch_counter) + '.jpg')
                s__patch_path = os.path.join(search_path, str(patch_counter) + '.jpg')

                line = t_patch_path + ',' + s__patch_path + ',' + str(scaled_gt.x1) + ',' + str(scaled_gt.y1) + ',' + \
                       str(scaled_gt.x2) + ',' + str(scaled_gt.y2)

                train_file.write(line)
            patch_counter += 1
