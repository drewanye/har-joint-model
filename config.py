__author__ = 'zhenanye'

def get_config():
    return HuynhConfig()

# config for Huynh Dataset
class HuynhConfig(object):

    def __init__(self):
        '''
        channels: the number of input sensor featuresn
        s_win_size: window length of simple activity
        c_win_size: the number of containing simple activities;
            so the window length of complex activity is  (s_win_size * c_win_size)
        s_labels_num: the number of simple activity labels
        c_labels_num: the number of complex activity labels
        batch_size: mini-batch size
        max_lr: max learning rate
        min_lr: min learning rate
        decay_speed: learning rate decay speed
        iter: iteration times
        dataset: data_set path
        test_point: the testing point
        '''
        self.channels = 12
        self.f_num = self.channels
        self.s_win_size = 50
        self.c_win_size = 15
        self.s_labels_num = 23
        self.c_labels_num = 4
        self.batch_size = 50
        self.norm = False
        self.max_lr = 0.0007
        self.min_lr = 0.0001
        self.decay_speed = 700
        self.iter = 7501
        self.dataset = "data/huynh.cp"
        self.test_point = 100
