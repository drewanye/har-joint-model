__author__ = 'zhenanye'

import tensorflow as tf
import numpy as np
# import complex_activity_model as cam
# import simple_activity_model as sam
import joint_model as jm
import load_data


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax

        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.strip().split(',')[:-1] for row in file
            ]]
        )
        file.close()
    # 3*27866*40

    return np.transpose(np.array(X_signals), (1, 2, 0)) # ** *40*3

def load_y(y_path, s_labels, c_labels):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    s_y = []
    c_y = []
    row_num = 0
    for r in file:
        row_num = row_num + 1
        pair = r.strip().split(',')
        s_y.append(s_labels.index(pair[0]))
        if row_num % 30 == 0:
            c_y.append(c_labels.index(pair[1]))
    # return simple labels and complex labels with int format
    return np.array(s_y, dtype=np.int32), np.array(c_y, dtype=np.int32)

# def load_y(y_path, s_labels, c_labels):
#     file = open(y_path, 'r')
#     # Read dataset from disk, dealing with text file's syntax
#     s_y = []
#     c_y = []
#     row_num = 0
#     for r in file:
#         row_num = row_num + 1
#         pair = r.strip().split(',')
#         s_y.append(s_labels.index(pair[0]))
#         c_y.append(c_labels.index(pair[1]))
#     # return simple labels and complex labels with int format
#     return np.array(s_y, dtype=np.int32), np.array(c_y, dtype=np.int32)

def one_hot(y_):
    #y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

if __name__=='__main__':

    # load data
    INPUT_SIGNAL_TYPES = [
        'ACC_X',
        'ACC_Y',
        'ACC_Z',
        'SW_X_40',
        'SW_Y_40',
        'SW_Z_40',
    ]

    SIMPLE_LABELS = [
        'Standing',
        'Walking',
        'Cycling',
        'Sitting',
        'Running',
        'Lying',
    ]

    COMPLEX_LABELS = [
        'Shoping',
        'Recreating',
        'Working',
        'Commuting',
        'Having a meal',
        'Sleeping',
        'House cleaning',
        'Meeting',
        'Exercise',
    ]

    DATA_PATH = 'data/'
    # DATASET_PATH = DATA_PATH + 'Pelab HAR Dataset/'
    DATASET_PATH = DATA_PATH + 'Huynh Dataset/'
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    # -----
    # load Pelab Har dataset
    # -----
    # simple_labels_path = DATASET_PATH + 'labels_simple.txt'
    # complex_labels_path = DATASET_PATH + 'labels_complex.txt'
    #
    # TRAIN = DATASET_PATH + 'train/'
    # TEST = DATASET_PATH + 'test/'
    #
    # X_train_dataset_paths = [TRAIN + i + '.txt' for i in INPUT_SIGNAL_TYPES]
    # X_test_dataset_paths = [TEST + j + '.txt' for j in INPUT_SIGNAL_TYPES]
    #
    # Y_train_dataset_path = TRAIN + 'labels.txt'
    # Y_test_dataset_path = TEST + 'labels.txt'
    #
    # # X's train and test set of simple activity
    # x_train = load_X(X_train_dataset_paths)  # 27120 * 40 * 6
    # x_test = load_X(X_test_dataset_paths)
    #
    # # X's train and test set of complex activity
    # x_c_train = np.reshape(x_train, (903, 1200, 6))
    # x_c_test = np.reshape(x_test, (398, 1200, 6))
    #
    # y_s_train, y_c_train = load_y(Y_train_dataset_path, SIMPLE_LABELS, COMPLEX_LABELS)
    # y_s_train = one_hot(y_s_train)
    # y_s_trains = np.reshape(y_s_train, (903, 30, 6))
    # y_c_train = one_hot(y_c_train)
    #
    # y_s_test, y_c_test = load_y(Y_test_dataset_path, SIMPLE_LABELS, COMPLEX_LABELS)
    # y_s_test = one_hot(y_s_test)
    # y_s_test_s = np.reshape(y_s_test, (398, 30, 6))
    # y_c_test = one_hot(y_c_test)

    # -----
    # load Huynh dataset
    # -----
    X_file_name = 'day{}-data.txt'
    YS_file_name = 'day{}-activity.txt'
    YC_file_name = 'day{}-routine.txt'
    DAYS = 7
    Features_num = 12
    s_labels_index = [1, 3, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 30, 33, 34, 35]

    X_INPUT_PATHS = [DATASET_PATH + X_file_name.format(i+1) for i in range(DAYS)]
    YS_INPUT_PATHS = [DATASET_PATH + YS_file_name.format(i+1) for i in range(DAYS)]
    YC_INPUT_PATHS = [DATASET_PATH + YC_file_name.format(i+1) for i in range(DAYS)]

    x = load_data.load_X(X_INPUT_PATHS)
    y_s = load_data.load_YS(YS_INPUT_PATHS)
    y_c = load_data.load_YC(YC_INPUT_PATHS)

    x_train, y_s_train, y_c_train, x_test, y_s_test, y_c_test = load_data.split_train_test(x, y_s, y_c)



    # ---------------
    # build and train convolution network model
    # ---------------

    # simple labels model
    # def run_simple_acti_model():
    #     X = tf.placeholder(dtype=tf.float32, shape=[None, 40, 6])
    #     Y = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    #     sim_acti_model = sam.SimpleActivityModel(X, Y, 903)
    #     pred_y = sim_acti_model.build_model()
    #     sim_acti_model.train_model(pred_y, x_train, y_s_train, x_test, y_s_test)
    #
    # # complex labels model
    # def run_complex_acti_model():
    #     X = tf.placeholder(dtype=tf.float32, shape=[None, 30*40, 6])
    #     Y = tf.placeholder(dtype=tf.float32, shape=[None, 9])
    #     # batch size 113 // 904 = 113 * 8
    #     com_acti_model = cam.ComplexActivityModel(X, Y, 100)
    #     pred_y = com_acti_model.build_model()
    #     com_acti_model.train_model(pred_y, x_c_train, y_c_train, x_c_test, y_c_test)

    # joint model
    def run_joint_model():
        # X = tf.placeholder(dtype=tf.float32, shape=[None, 30*40, 6])
        # Y = tf.placeholder(dtype=tf.float32, shape=[None, 9])
        # Ys = tf.placeholder(dtype=tf.float32, shape=[None, 30, 6])
        X = tf.placeholder(dtype=tf.float32, shape=[None, 750, 12])
        Y = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        Ys = tf.placeholder(dtype=tf.float32, shape=[None, 15, 23])
        j_model = jm.JointModel(X, Y, Ys, 219)
        j_model.build_model()
        j_model.train_model(x_train, y_c_train, y_s_train, x_test, y_c_test, y_s_test)

    run_joint_model()
    # run_simple_acti_model()
    # run_complex_acti_model()






