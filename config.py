__author__ = 'zhenanye'
import data
import data1

def get_config():
    # return OpportunityConfig()
    return Config()
# config for Pelab
# class Config(object):
#
#
#     def __init__(self):
#
#         self.f_num = 8
#         self.s_win_size = 40
#         self.c_win_size = 30
#         self.s_labels_num = 6
#         self.c_labels_num = 9
#         self.batch_size = 50
#         self.norm = False
#         # learning rate
#         self.max_lr = 0.003
#         self.min_lr = 0.0001
#         self.decay_speed = 200
#         self.iter = 2101
#         self.dataset = data.PelabDataset("User", self)

class HuynhConfig(object):
    '''This Config class is used to test'''

    def __init__(self):
        self.channels = 12
        self.overlap_ratio = 0.5
        self.s_win_size = 50
        self.c_win_size = 15
        # self.s_labels_num = 25
        self.s_labels_num = 36
        # self.c_labels_num = 4
        self.c_labels_num = 5
        self.batch_size = 50
        self.norm = False
        self.max_lr = 0.003
        self.min_lr = 0.0001
        self.decay_speed = 200
        self.iter = 2101
        self.test_point = 20

# config for Huynh of PLY providing
class Config(object):

    def __init__(self):
        self.channels = 12
        self.f_num = self.channels
        self.s_win_size = 50
        self.c_win_size = 15
        # self.s_labels_num = 25
        self.s_labels_num = 23
        # self.c_labels_num = 4
        self.c_labels_num = 4
        self.batch_size = 50
        self.norm = False
        self.max_lr = 0.0007
        self.min_lr = 0.0001
        self.decay_speed = 700
        self.iter = 7501
        # self.dataset = data.HuynhOriginalDataset("HuynhWithoutUnlabeled/", self)
        self.dataset = data1.HuynhDataset("Huynh Dataset/", self)
        self.test_point = 100

class OpportunityConfig(object):
    def __init__(self):
        ''' Without unlabeled data
        '''
        self.channels = 113
        self.overlap_ratio = 0.5
        self.s_win_size = 24
        self.c_win_size = 10  # 10
        self.s_labels_num = 4
        self.c_labels_num = 5
        self.batch_size = 64
        self.norm = False
        self.max_lr = 0.003
        self.min_lr = 0.0001
        self.decay_speed = 700
        self.iter = 2101
        self.test_point = 30

# config for opportunity dataset
# class Config(object):
#
#     def __init__(self):
#         self.f_num = 242
#         self.s_win_size = 60
#         self.c_win_size = 10
#         # self.s_labels_num = 17
#         self.s_labels_num = 4
#         # self.c_labels_num = 4
#         self.c_labels_num = 5
#         self.batch_size = 50
#         self.norm = False
#         self.max_lr = 0.01
#         self.min_lr = 0.001
#         self.decay_speed = 700
#         self.iter = 3301
#         # self.dataset = data.OpportunityDataset("Opportunity", self)
#         self.dataset = data.OpportunityDataset("Opportunity_Locomotion", self)

# config for opportunity 113 dataset
# class Config(object):
#
#     def __init__(self):
#         self.f_num = 113
#         self.s_win_size = 24
#         self.overlap_ratio = 0.5
#         self.c_win_size = 15
#         # self.s_labels_num = 17
#         self.s_labels_num = 18
#         # self.c_labels_num = 4
#         self.c_labels_num = 6
#         self.s_tasks = (self.c_win_size*self.s_win_size-self.s_win_size) / int(self.s_win_size*self.overlap_ratio) + 1
#         self.batch_size = 50
#         self.norm = False
#         self.max_lr = 0.03
#         self.min_lr = 0.001
#         self.decay_speed = 700
#         self.iter = 3301
#         # self.dataset = data.OpportunityDataset("Opportunity", self)
#         self.dataset = data.Opportunity113Dataset("opportunity.cp", self)
