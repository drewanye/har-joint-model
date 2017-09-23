__author__ = 'zhenanye'

def get_config():
    return HuynhConfig()

# config for Huynh Dataset
class HuynhConfig(object):

    def __init__(self):
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
