__author__ = 'zhenanye'

import tensorflow as tf
import utils
import numpy as np
import math
import os
from sklearn import utils as skutils
import cPickle as cp


class SimpleActivity:

    def __init__(self, X, Y, cfg, is_training, norm=False):
        self.X = X
        self.Y = Y
        self.batch_size = cfg.batch_size
        self.is_training = is_training
        self.norm = norm
        self.labels_num = cfg.s_labels_num

    def _add_layers(self, x):
        def residual(x, in_channel, out_channel, is_training, norm):
            """residual unit with 2 layers
            convolution:
                width filter: 1
                height filter: 3
            """
            orig_x = x
            with tf.variable_scope('conv1'):
                conv1 = utils.conv(x, [1, 3, in_channel, out_channel], [out_channel], padding='SAME')
                if norm:
                    conv1 = utils.batch_norm(conv1, is_training)
                relu1 = utils.activation(conv1)
            with tf.variable_scope('conv2'):
                conv2 = utils.conv(relu1, [1, 3, out_channel, out_channel], [out_channel], padding='SAME')
                if norm:
                    conv2 = utils.batch_norm(conv2, is_training)
            with tf.variable_scope('add'):
                if in_channel != out_channel:
                    orig_x = utils.conv(x, [1, 1, in_channel, out_channel], [out_channel], padding='SAME')

            return utils.activation(conv2 + orig_x)

        x_shape = x.get_shape()
        with tf.variable_scope('residual1'):
            r1 = residual(x, x_shape[-1], 32, self.is_training, self.norm)
            tf.summary.histogram('res_output1', r1)
        with tf.variable_scope('residual2'):
            r2 = residual(r1, r1.get_shape()[-1], 32, self.is_training, self.norm)
            tf.summary.histogram('res_output2', r2)

        with tf.variable_scope('pool0'):
            h_pool0 = utils.max_pool(r2, 1, 2, 1, 2, padding='SAME')

        with tf.variable_scope('residual3'):
            r3 = residual(h_pool0, h_pool0.get_shape()[-1], 64, self.is_training, self.norm)
            tf.summary.histogram('res_output3', r3)
        with tf.variable_scope('residual4'):
            r4 = residual(r3, r3.get_shape()[-1], 64, self.is_training, self.norm)
            tf.summary.histogram('res_output4', r4)

        with tf.variable_scope('pool1'):
            h_pool1 = utils.max_pool(r4, 1, 5, 1, 5, padding='SAME')

        with tf.variable_scope('full_conn_1'):
            flat_size = 5 * 64
            h_pool2_flat = tf.reshape(h_pool1, [-1, flat_size])
            h_fc1 = utils.full_conn(h_pool2_flat, [flat_size, 1024], [1024])
            h_fc1 = utils.activation(h_fc1)

        with tf.variable_scope('full_conn_2'):
            h_fc2 = utils.full_conn(h_fc1, [1024, 128], [128])
            h_fc2 = utils.activation(h_fc2)
        return h_fc2

    def build_model(self):
        x = utils.input_batch_norm(self.X)
        h_fc1 = self._add_layers(x)

        concat_outputs = h_fc1
        with tf.variable_scope('scores'):
            pred_y = utils.scores(h_fc1, [128, self.labels_num], [self.labels_num])

        with tf.variable_scope('train'):
            lambda_loss_amount =0.0015
            l2 = lambda_loss_amount * \
                 sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_y)) + l2
            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(pred_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return concat_outputs, cross_entropy, accuracy, correct_prediction

class JointModel(object):

    def __init__(self, X, YS, YC, cfg, log_path, version=""):

        self.X = X
        self.YC = YC
        self.YS = YS
        self.config = cfg
        # sensor channels
        self.channels = cfg.channels
        # simple activity window size
        self.s_win_size = cfg.s_win_size
        # complex activity window size
        self.c_win_size = cfg.c_win_size
        self.batch_size = cfg.batch_size
        self.s_labels_num = cfg.s_labels_num
        self.c_labels_num = cfg.c_labels_num
        self.pos = 0
        self.pos_test = 0
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.log_path = log_path
        self.norm = cfg.norm
        self.prok = tf.placeholder(dtype=tf.float32)
        self.train_saved_dir = os.path.join("train", version)

    def load_data(self, test_day):
        print("Loading data.............")
        f = file(self.config.dataset, 'rb')
        data = cp.load(f)
        f.close()
        data = data[test_day]
        x_train, y_s_train, y_c_train = data[0]
        x_test, y_s_test, y_c_test = data[1]
        self.x_train, self.y_s_train, self.y_c_train = skutils.shuffle(x_train, y_s_train, y_c_train, random_state=0)
        self.x_test, self.y_s_test, self.y_c_test = skutils.shuffle(x_test, y_s_test, y_c_test, random_state=0)
        print("Train and Test data shape:")
        print("x_train: {} ".format(self.x_train.shape) + \
            "y_s_train: {} ".format(self.y_s_train.shape) + \
            "y_c_train: {} ".format(self.y_c_train.shape) + \
            "x_test: {} ".format(self.x_test.shape) + \
            "y_s_test: {} ".format(self.y_s_test.shape) +\
            "y_c_test: {}".format(self.y_c_test.shape)
        )
        # set different log for diff test_days
        self.log_path = self.log_path + "test{}/".format(test_day)

    def next_batch(self):
        train_size = self.x_train.shape[0]
        scale = self.pos+self.batch_size
        if scale > train_size:
            a = scale-train_size
            x1 = self.x_train[self.pos:]
            x2 = self.x_train[0:a]
            y_c1 = self.y_c_train[self.pos:]
            y_c2 = self.y_c_train[0:a]
            y_s1 = self.y_s_train[self.pos:]
            y_s2 = self.y_s_train[0:a]
            self.pos = a
            return np.concatenate((x1, x2)), np.concatenate((y_c1, y_c2)), np.concatenate((y_s1, y_s2))
        else:
            x = self.x_train[self.pos:scale]
            y_c = self.y_c_train[self.pos:scale]
            y_s = self.y_s_train[self.pos:scale]
            self.pos = scale
            return x, y_c, y_s

    def build_model(self):
        x_serie_c = self.X
        xs_s = tf.split(x_serie_c, num_or_size_splits=self.config.c_win_size, axis=1)
        ys_s = tf.split(self.YS, num_or_size_splits=self.config.c_win_size, axis=1)
        concat_outputs = []
        self.losses = []
        self.accuracies = []
        self.correct_preds = []
        with tf.variable_scope('simple_activity') as scope:
            is_reuse = False
            for i, j in zip(xs_s, ys_s):
                sa = SimpleActivity(i, tf.reshape(j, [-1, self.s_labels_num]), self.config,
                                    is_training=self.is_training, norm=self.norm)
                output, loss, accuracy, correct_pred_s = sa.build_model()
                concat_outputs.append(output)
                self.losses.append(loss)
                self.accuracies.append(accuracy)
                self.correct_preds.append(correct_pred_s)
                if not is_reuse:
                    scope.reuse_variables()
                    is_reuse = True

            self.s_mean_loss = tf.reduce_mean(self.losses)
            tf.summary.scalar('loss', self.s_mean_loss)
            self.s_mean_accuracy = tf.reduce_mean(self.accuracies)
            tf.summary.scalar('accuracy', self.s_mean_accuracy)
        self.train_step_s = tf.train.AdamOptimizer(self.learning_rate).minimize(self.s_mean_loss)
        with tf.variable_scope('complex_activity'):
            with tf.variable_scope("lstm_layers"):
                lstm_size = 128
                cells = tf.contrib.rnn.MultiRNNCell([utils.lstm_cell(lstm_size) for _ in range(3)], state_is_tuple=True)
                outputs, states = tf.contrib.rnn.static_rnn(cells, concat_outputs, dtype=tf.float32)

            pred_y_c = utils.scores(outputs[-1], [lstm_size, self.c_labels_num], [self.c_labels_num])
            lambda_loss_amount =0.0015
            l2 = lambda_loss_amount * \
                 sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.YC, logits=pred_y_c)) \
                                + l2
            tf.summary.scalar("loss", cross_entropy)
            self.train_step_c = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
            tf.summary.scalar("learning_rate", self.learning_rate)

            self.joint_loss = cross_entropy + self.s_mean_loss
            self.c_loss= cross_entropy
            tf.summary.scalar("joint_loss", self.joint_loss)

            self.joint_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.joint_loss)
            self.correct_prediction_c = tf.equal(tf.argmax(self.YC, 1), tf.argmax(pred_y_c, 1))
            self.c_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction_c, tf.float32))
            tf.summary.scalar("accuracy", self.c_accuracy)

    def train_model(self):
        max_lr = self.config.max_lr
        min_lr = self.config.min_lr
        decay_speed = self.config.decay_speed
        s_best_accuracy = 0.0
        c_best_accuracy = 0.0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        # tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # tf_config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_path + '/train',
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(self.log_path + '/test')
            sess.run(tf.global_variables_initializer())
            for i in range(self.config.iter):
                lr = min_lr + (max_lr - min_lr) * math.exp(-i / decay_speed)
                xt, yc, ys = self.next_batch()
                _, summary = sess.run([self.joint_train_step, merged],

                         feed_dict={
                             self.X: xt,
                             self.YC: yc,
                             self.YS: ys,
                             self.learning_rate: lr,
                             self.is_training: True,
                             self.prok: 0.5
                                    })
                train_writer.add_summary(summary, i)
                if i%self.config.test_point==0:
                    s_accs, s_mean_acc, s_lss, c_acc, c_ls, j_loss, t_summary = sess.run([
                                                            self.accuracies,
                                                            self.s_mean_accuracy,
                                                            self.losses,
                                                            self.c_accuracy,
                                                            self.c_loss,
                                                            self.joint_loss,
                                                            merged,
                    ],
                                    feed_dict={
                                        self.X: self.x_test,
                                        self.YC: self.y_c_test,
                                        self.YS: self.y_s_test,
                                        self.learning_rate: lr,
                                        self.is_training: False,
                                        self.prok: 1.0
                                    })
                    test_writer.add_summary(t_summary, i)
                    s_best_accuracy = max(s_best_accuracy, max(s_accs))
                    c_best_accuracy = max(c_acc, c_best_accuracy)
                    print("-----Iteration: {}-----".format(i))
                    print("-----Simple Activiy-----")
                    print("mean_accuracy:{}".format(s_mean_acc))
                    print("acurracies: {}".format(s_accs))
                    print("loss: {}".format(s_lss))
                    print("-----Complex Activity-----")
                    print("acurracy:{}".format(c_acc))
                    print("loss:{}".format(c_ls))
                    print("joint_loss: {}".format(j_loss))
                    print("")
            print("-----Final Report-----")
            print(">>> Simple Activity")
            print("final test accuracy: {}".format(s_accs))
            print("best iteration's test accuracy: {}".format(s_best_accuracy))
            print(">>> Complex Activity")
            print("final test accuracy: {}".format(c_acc))
            print("best iteration's test accuracy: {}".format(c_best_accuracy))





