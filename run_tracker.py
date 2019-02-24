import cv2
import sys
import itertools
import tensorflow as tf
from multiprocessing import Pool
from random import randint

from bounding_box import BoundingBox
import goturn_net
from frame_manager import FrameManager


def _run_tracker(data):
    frame_manager, frame, sess = data
    prev_img = frame_manager.get_image()
    frame_manager.update_frame(frame)
    current_img = frame_manager.get_image()
    print('HERE RUN1')
    prediction = sess.run(goturn.fc4, feed_dict={goturn.image: [current_img],
                                                 goturn.target: [prev_img],
                                                 }
                          )
    print('HERE RUN 2')
    prediction_scaled = prediction[0] / 10
    frame_manager.update_bbox(prediction_scaled)
    print('REACHED THE END OF RUN TRACKER METHOD')
    return frame_manager.predicted_bbox


def draw_bboxes(frame, predicted_bboxes):
    for predicted_bbox in predicted_bboxes:
        p1 = (int(predicted_bbox.x1), int(predicted_bbox.y1))
        p2 = (int(predicted_bbox.x2), int(predicted_bbox.y2))
        cv2.rectangle(frame, p1, p2, (randint(0, 255), randint(0, 255), randint(0, 255)), 2, 1)
    return frame

if __name__ == '__main__':
    video = cv2.VideoCapture('train_video.avi')
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')

    bboxes = cv2.selectROIs('MULTI TRACKING VERSION', frame)
    init_rois = [
        BoundingBox(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        for bbox in bboxes
    ]
    frame_managers = [
        FrameManager(frame, init_roi)
        for init_roi in init_rois
    ]
    print(bboxes)
    goturn = goturn_net.TRACKNET(batch_size=1, train=False)
    goturn.build()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt_dir = './checkpoints'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    pool = Pool(len(bboxes))

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        print('HERE')
        pool = Pool(len(bboxes))
        predicted_bboxes = pool.map(
            _run_tracker,
            [(frame_manager, frame, sess) for frame_manager in frame_managers]
        )
        pool.close()
        pool.join()
        print('PRED BBOXES', predicted_bboxes)
        frame_with_pred = draw_bboxes(predicted_bboxes, frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame_with_pred, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)
        cv2.imshow('GOTURN TRACKER', frame_with_pred)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

