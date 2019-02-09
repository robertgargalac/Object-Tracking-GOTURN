import cv2
import sys
import tensorflow as tf

from bounding_box import BoundingBox
import goturn_net
from frame_manager import FrameManager

print('before video loading')
video = cv2.VideoCapture('exp_avion.avi')
print('after')
if not video.isOpened():
    print('Could not open video')
    sys.exit()
print('after2')
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
man_bbox = cv2.selectROI(frame, False)
roi = BoundingBox(man_bbox[0], man_bbox[1], man_bbox[0] + man_bbox[2], man_bbox[1] + man_bbox[3])
frame_manager = FrameManager(frame, roi)
img = frame_manager.get_image()

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
    print(prediction_scaled)
    # cv2.imshow('prev_img', prev_img)
    # cv2.waitKey(1000)
    # cv2.imshow('current_img', current_img)
    # cv2.waitKey(1000)
    cv2.imshow('GOTURN TRACKER', frame_manager.frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break