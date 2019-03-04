import cv2
from random import randint


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