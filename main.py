__author__ = 'zhenanye'

import tensorflow as tf
import joint_model as jm
import argparse
import config

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Set the test day to split train and test data")
    parser.add_argument('--test', type=int, default=0, help='Select the test day. Max num is 6')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--gpu', type=int, default=0, help='assign task to selected gpu')
    args = parser.parse_args()

    LOGS = "logs/"
    log_path = LOGS + args.version + '/'
    # joint model for Huynh dataset
    def run_joint_model():
        cfg = config.get_config()
        cfg.gpu = args.gpu
        X = tf.placeholder(dtype=tf.float32, shape=[None, cfg.c_win_size, cfg.s_win_size, cfg.channels])
        YC = tf.placeholder(dtype=tf.float32, shape=[None, cfg.c_labels_num])
        YS = tf.placeholder(dtype=tf.float32, shape=[None, cfg.c_win_size, cfg.s_labels_num])
        model = jm.JointModel(X, YS, YC, cfg, log_path, args.version)
        # load data
        model.load_data(args.test)
        model.build_model()
        model.train_model()

    run_joint_model()






