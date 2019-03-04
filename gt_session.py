import goturn_net
import tensorflow as tf


class GtSession:
    def __init__(self, i):
        self.goturn = goturn_net.TRACKNET(batch_size=1, train=False)
        self.goturn.build()

        self.start_session(i)

    def start_session(self, i):
        self.sess = tf.Session()
        saver = tf.train.Saver()

        ckpt_dir = './checkpoints'
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Session number {} was created successfully'.format(i))
