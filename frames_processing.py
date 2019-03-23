import cv2
import os
from collections import defaultdict
import xml.etree.ElementTree as ET

from bounding_box import BoundingBox
from utils import get_patch

train_target_folder = 'train\\target'
train_search_folder = 'train\\search'

data_folder = 'videos'
ann_folder = 'ann'

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

file_name = '01-Light_video00001.xml'
document = ET.parse(file_name)
root = document.getroot()
frame_content = cv2.imread('00000001.jpg')
next_frame_content = cv2.imread('')
for index, frame in enumerate(root):
    target_gt = BoundingBox()
    search_gt = BoundingBox()

    a_point_target = frame.find('A')
    c_point_target = frame.find('C')

    if not a_point_target and not c_point_target:
        continue

    a_point_search = frame[index + 1].find('A')
    c_point_search = frame[index + 1].find('C')

    x2_t = float(a_point_target.find('x').text)
    y1_t = float(a_point_target.find('y').text)
    x1_t = float(c_point_target.find('x').text)
    y2_t = float(c_point_target.find('y').text)

    x2_s = float(a_point_search.find('x').text)
    y1_s = float(a_point_search.find('y').text)
    x1_s = float(c_point_search.find('x').text)
    y2_s = float(c_point_search.find('y').text)

    target_gt .update_coordinates(x1_t, x2_t, y1_t, y2_t)
    search_gt.update_coordinates(x1_s, x2_s, y1_s, y2_s)

    target = get_patch(frame_content, target_gt)
    search = get_patch(next_frame_content, target_gt)

    cv2.imwrite(os.path.join(target_path, str(index - 2) + '.jpg'), target)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(search_path, str(index - 2) + '.jpg'), search)
    cv2.waitKey(0)
