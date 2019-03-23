import cv2
from random import randint

from bounding_box import BoundingBox


def run_tracker(data):
    frame_manager, gt_sess, frame = data

    prev_img = frame_manager.get_image()
    frame_manager.update_frame(frame)
    current_img = frame_manager.get_image()

    prediction = gt_sess.sess.run(gt_sess.goturn.fc4, feed_dict={
        gt_sess.goturn.image: [current_img],
        gt_sess.goturn.target: [prev_img],
    }
                                  )
    prediction_scaled = prediction[0] / 10
    frame_manager.update_bbox(prediction_scaled)
    return frame_manager.predicted_bbox


def draw_bboxes(frame, predicted_bboxes):
    for predicted_bbox in predicted_bboxes:
        p1 = (int(predicted_bbox.x1), int(predicted_bbox.y1))
        p2 = (int(predicted_bbox.x2), int(predicted_bbox.y2))
        cv2.rectangle(frame, p1, p2, (randint(0, 255), randint(0, 255), randint(0, 255)), 2, 1)
    return frame


def get_patch(frame, bbox):
    frame_h, frame_w, _ = frame.shape

    x_center = bbox.get_x_center()
    y_center = bbox.get_y_center()
    image_h = bbox.compute_output_height()
    image_w = bbox.compute_output_width()

    y1_img = max(1, int(y_center - (image_h / 2)))
    y2_img = int(y_center + (image_h / 2))
    x1_img = max(1, int(x_center - (image_w / 2)))
    x2_img = int(x_center + (image_w / 2))

    if y2_img >= frame_h:
        y2_img = frame_h
    if x2_img >= frame_w:
        x2_img = frame_w

    image = frame[
             y1_img: y2_img,
             x1_img: x2_img
            ]
    return image


def scale_gt(target_bbox, search_bbox, target):
    target_h, target_w, _ = target.shape

    x1_diff = search_bbox.x1 - target_bbox.x1
    x2_diff = search_bbox.x2 - target_bbox.x2
    y1_diff = search_bbox.y1 - target_bbox.y1
    y2_diff = search_bbox.y2 - target_bbox.y2

    x1_diff_scaled = x1_diff / target_w
    x2_diff_scaled = x2_diff / target_w
    y1_diff_scaled = y1_diff / target_h
    y2_diff_scaled = y2_diff / target_h

    x1_gt = 0.25 + x1_diff_scaled
    x2_gt = 0.75 + x2_diff_scaled
    y1_gt = 0.25 + y1_diff_scaled
    y2_gt = 0.75 + y2_diff_scaled

    scaled_bbox = BoundingBox(x1_gt, y1_gt, x2_gt, y2_gt)
    return scaled_bbox
