import cv2
import sys
import time
import tensorflow as tf

from bounding_box import BoundingBox
import goturn_net
from frame_manager import FrameManager

video = cv2.VideoCapture('train_video.avi')
if not video.isOpened():
    print('Could not open video')
    sys.exit()
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

man_bbox = cv2.selectROI(frame, False)
roi = BoundingBox(man_bbox[0], man_bbox[1], man_bbox[0] + man_bbox[2], man_bbox[1] + man_bbox[3])
frame_manager = FrameManager(frame, roi)

goturn = goturn_net.TRACKNET(batch_size=1, train=False)
goturn.build()
sess = tf.Session()
saver = tf.train.Saver()
ckpt_dir = './checkpoints'
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
saver.restore(sess, ckpt.model_checkpoint_path)

while True:
    ok, frame = video.read()
    if not ok:
        break
    timer = cv2.getTickCount()
    prev_img = frame_manager.get_image()
    frame_manager.update_frame(frame)
    current_img = frame_manager.get_image()

    prediction = sess.run(goturn.fc4, feed_dict={goturn.image: [current_img],
                                                 goturn.target: [prev_img],
                                                 }
                          )
    prediction_scaled = prediction[0] / 10

    frame_manager.update_bbox(prediction_scaled)
    frame_manager.draw_bbox()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (50, 170, 50), 2)
    cv2.imshow('GOTURN TRACKER', frame_manager.frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break