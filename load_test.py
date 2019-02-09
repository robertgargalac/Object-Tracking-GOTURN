import cv2
import goturn_net
import matplotlib.pyplot as plt

import tensorflow as tf

current_img = cv2.imread('search1.jpg')
prev_img = cv2.imread('target1.jpg')
current_img_res = cv2.resize(current_img, (227, 227))
prev_img_res = cv2.resize(prev_img, (227, 227))
print(current_img_res.shape)
print(prev_img_res.shape)

goturn = goturn_net.TRACKNET(batch_size=1, train=False)
goturn.build()
sess = tf.Session()
saver = tf.train.Saver()
ckpt_dir = './checkpoints'
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
print(ckpt)
saver.restore(sess, ckpt.model_checkpoint_path)

predicted_bbox = sess.run(goturn.fc4, feed_dict={goturn.image: [current_img_res],
                                                 goturn.target: [prev_img_res],
                                                 }
                          )
print(predicted_bbox)
bbox = (predicted_bbox[0] / 10)
print(bbox)
h, w, _ = current_img.shape
p1 = (int(bbox[0] * w), int(bbox[1] * h))
p2 = (int(bbox[2] * w), int(bbox[3] * h))
cv2.rectangle(current_img, p1, p2, (255,0,0), 2, 1)
plt.imshow(current_img)
plt.show()

