import tensorflow as tf


class TestSoftMarginTripletLoss(tf.test.TestCase):

    def setUp(self):
        super(TestSoftMarginTripletLoss).setUp()

    def testSoftMarginLoss(self):
        dist_pos = tf.Tensor()
