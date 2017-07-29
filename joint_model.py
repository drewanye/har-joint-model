__author__ = 'zhenanye'


import tensorflow as tf
import utils

class SimpleActivity:

    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def build_model(self):
        with tf.variable_scope('conv1'):
            relu1 = utils.conv_relu(self.X, [1, 3, 12, 32], [32])
            # import pdb;pdb.set_trace()
        with tf.variable_scope('conv2'):
            #import pdb;pdb.set_trace()
            relu2 = utils.conv_relu(relu1, [1, 3, 32, 64], [64])
            concat_outputs = relu2
            h_pool2 = utils.max_pool(relu2, 1, 2, 1, 2)
        with tf.variable_scope('full_conn_1'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 25*64])
            h_fc1 = utils.full_conn_relu(h_pool2_flat, [25*64, 1024], [1024])
        with tf.variable_scope('scores'):
            pred_y = utils.scores(h_fc1, [1024, 23], [23])

        with tf.variable_scope('train'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_y))
            # print("**** cross_entropy name: {}".format(self.cross_entropy.name))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            # print("**** train_step name: {}".format(self.train_step.name))
            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(pred_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print("**** accuracy name: {}".format(self.accuracy.name))

        return concat_outputs, cross_entropy, train_step, accuracy


class JointModel:

    # def __init__(self, X1, Y1, X2, Y2, batch_size1, batch_size2):
    #
    #     self.X1 = X1
    #     self.Y1 = Y1
    #     self.X2 = X2
    #     self.Y2 = Y2
    #     self.batch_size1 = batch_size1
    #     self.batch_size2 = batch_size2

    def __init__(self, X, Y, Y2, batch_size):

        self.X = X
        self.Y = Y
        self.Y2 = Y2
        self.batch_size = batch_size

    def build_model(self):
        x_serie_c = tf.reshape(self.X, [-1, 15, 50, 12])
        xs_s = tf.split(x_serie_c, num_or_size_splits=15, axis=1)
        ys_s = tf.split(self.Y2, num_or_size_splits=15, axis=1)
        concat_outputs = []
        self.losses = []
        self.train_steps = []
        self.accuracies = []
        w_n = len(xs_s)
        with tf.variable_scope('simple_activity') as scope:
            is_reuse = False
            for i, j in zip(xs_s, ys_s):
                sa = SimpleActivity(i, tf.reshape(j, [-1, 23]), self.batch_size)
                output, loss, step, accuracy = sa.build_model()
                concat_outputs.append(output)
                self.losses.append(loss)
                self.train_steps.append(step)
                self.accuracies.append(accuracy)
                if not is_reuse:
                    scope.reuse_variables()
                    is_reuse = True

        loss_weight = tf.constant(1.0/w_n, dtype=tf.float32)
        self.s_total_losses = tf.multiply(loss_weight, tf.reduce_sum(self.losses))
        self.train_step_s = tf.train.AdamOptimizer(1e-4).minimize(self.s_total_losses)
        with tf.variable_scope('complex_activity'):
            with tf.variable_scope('pre_conv1'):
                pre_relu1 = utils.conv_relu(x_serie_c, [3, 1, 12, 32], [32])
                pre_pool1 = utils.max_pool(pre_relu1, 3, 1, 3, 1)
            with tf.variable_scope('pre_conv2'):
                pre_relu2 = utils.conv_relu(pre_pool1, [3, 1, 32, 64], [64])
                pre_pool2 = utils.max_pool(pre_relu2, 5, 1, 5, 1)
            x_serie_c = tf.concat([pre_pool2]+concat_outputs, 3) #  -1, 1, 40, 128

            with tf.variable_scope('conv1'):
                relu1 = utils.conv_relu(x_serie_c, [1, 3, 15*64+64, 32], [32])
                h_pool1 = utils.max_pool(relu1, 1, 2, 1, 2)
            with tf.variable_scope('conv2'):
                relu2 = utils.conv_relu(h_pool1, [1, 3, 32, 64], [64])
                h_pool2 = utils.max_pool(relu2, 1, 5, 1, 5)
            with tf.variable_scope('full_conn_1'):
                h_pool2_flat = tf.reshape(h_pool2, [-1, 5*64])
                h_fc1 = utils.full_conn_relu(h_pool2_flat, [5*64, 1024], [1024])

            pred_y_c = utils.scores(h_fc1, [1024, 4], [4])

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_y_c))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
            correct_prediction_c = tf.equal(tf.argmax(self.Y, 1), tf.argmax(pred_y_c, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction_c, tf.float32))
        self.train_steps.append(self.train_step)
        self.losses.append(self.cross_entropy)
        self.accuracies.append(self.accuracy)

    def train_model(self, x_train, y_train, y_strain_s, x_test, y_test, y_test_s):
        train_size = len(x_train)
        batch_size = self.batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(301):
                for start, end in zip(range(0, train_size, batch_size),
                                      range(batch_size, train_size+1,  batch_size)):
                    sess.run([self.train_step_s, self.train_step],
                             feed_dict={
                                 self.X: x_train[start:end],
                                 self.Y: y_train[start:end],
                                 self.Y2: y_strain_s[start:end]
                                        })
                if i%30==0:
                    accs, lss = sess.run([self.accuracies, self.losses],
                             feed_dict={
                                 self.X: x_test,
                                 self.Y: y_test,
                                 self.Y2: y_test_s,
                            })
                    print("-----Simple Activiy-----")
                    print("acurracies: {}".format(accs[:-1]))
                    print("losses: {}".format(lss[:-1]))
                    print("-----Complex Activity-----")
                    print("acurracy:{}".format(accs[-1]))
                    print("losses:{}".format(lss[-1]))

    # def build_model(self):
    #     x_serie_c = tf.reshape(self.X, [-1, 30, 40, 6])
    #     xs_s = tf.split(x_serie_c, num_or_size_splits=30, axis=1)
    #     ys_s = tf.split(self.Y2, num_or_size_splits=30, axis=1)
    #     concat_outputs = []
    #     self.losses = []
    #     self.train_steps = []
    #     self.accuracies = []
    #     w_n = len(xs_s)
    #     with tf.variable_scope('simple_activity') as scope:
    #         is_reuse = False
    #         for i, j in zip(xs_s, ys_s):
    #             sa = SimpleActivity(i, tf.reshape(j, [-1, 6]), self.batch_size)
    #             output, loss, step, accuracy = sa.build_model()
    #             concat_outputs.append(output)
    #             self.losses.append(loss)
    #             self.train_steps.append(step)
    #             self.accuracies.append(accuracy)
    #             if not is_reuse:
    #                 scope.reuse_variables()
    #                 is_reuse = True
    #
    #     loss_weight = tf.constant(1.0/w_n, dtype=tf.float32)
    #     self.s_total_losses = tf.multiply(loss_weight, tf.reduce_sum(self.losses))
    #     self.train_step_s = tf.train.AdamOptimizer(1e-4).minimize(self.s_total_losses)
    #     with tf.variable_scope('complex_activity'):
    #         with tf.variable_scope('pre_conv1'):
    #             pre_relu1 = utils.conv_relu(x_serie_c, [3, 1, 6, 32], [32])
    #             pre_pool1 = utils.max_pool(pre_relu1, 3, 1, 3, 1)
    #         with tf.variable_scope('pre_conv2'):
    #             pre_relu2 = utils.conv_relu(pre_pool1, [3, 1, 32, 64], [64])
    #             pre_pool2 = utils.max_pool(pre_relu2, 10, 1, 10, 1)
    #         x_serie_c = tf.concat([pre_pool2]+concat_outputs, 3) #  -1, 1, 40, 128
    #
    #         with tf.variable_scope('conv1'):
    #             relu1 = utils.conv_relu(x_serie_c, [1, 3, 30*64+64, 32], [32])
    #             h_pool1 = utils.max_pool(relu1, 1, 2, 1, 2)
    #         with tf.variable_scope('conv2'):
    #             relu2 = utils.conv_relu(h_pool1, [1, 3, 32, 64], [64])
    #             h_pool2 = utils.max_pool(relu2, 1, 2, 1, 2)
    #         with tf.variable_scope('full_conn_1'):
    #             h_pool2_flat = tf.reshape(h_pool2, [-1, 10*64])
    #             h_fc1 = utils.full_conn_relu(h_pool2_flat, [10*64, 1024], [1024])
    #
    #         pred_y_c = utils.scores(h_fc1, [1024, 9], [9])
    #
    #         self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred_y_c))
    #         self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    #         correct_prediction_c = tf.equal(tf.argmax(self.Y, 1), tf.argmax(pred_y_c, 1))
    #         self.accuracy = tf.reduce_mean(tf.cast(correct_prediction_c, tf.float32))
    #     self.train_steps.append(self.train_step)
    #     self.losses.append(self.cross_entropy)
    #     self.accuracies.append(self.accuracy)



