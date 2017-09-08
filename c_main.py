__author__ = 'zhenanye'

import tensorflow as tf
import complex_activity as cam
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Set the test day to split train and test data")
    parser.add_argument('--test_day', type=int, default=0, help='Select the test day. Max num is 6')
    args = parser.parse_args()

    # mono model for Huynh dataset
    def run_complex_activity_model():
        X = tf.placeholder(dtype=tf.float32, shape=[None, 750, 12])
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        batch_size = 50
        c_model = cam.ComplexActivityModel(X, Y, batch_size)
        # load data
        c_model.load_data(args.test_day)
        c_model.build_model()
        c_model.train_model()

    run_complex_activity_model()






