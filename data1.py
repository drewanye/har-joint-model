__author__ = 'zhenanye'


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import cPickle as cp
import utils

def one_hot(y_, n_values):
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

DATA_PATH = 'data/'

class DataSet(object):

    def __init__(self, dataset, cfg):
        self.dataset_path = DATA_PATH + dataset
        self.config = cfg

    def _load_X(self, inputs):
        raise NotImplementedError

    def _load_YS(self, inputs):
        raise NotImplemented

    def _load_YC(self, inputs):
        raise NotImplementedError

    def get_train_test(self, test_one):
        raise NotImplementedError

# -------
# load Huynh dataset
# -------

class HuynhDataset(DataSet):

    X_file_name = 'day{}-data.txt'
    YS_file_name = 'day{}-activity.txt'
    YC_file_name = 'day{}-routine.txt'
    DAYS = 7
    Features_num = 12
    s_labels_index = [1, 3, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 30, 33, 34, 35]

    def _load_X(self, inputs):
        res = []
        for i in inputs:
            with open(i, 'r') as file:
                for r in file:
                    res.append(r.strip().split(',')[:-1])
        return np.reshape(res, (len(res)/750, 15, 50, self.config.f_num))
        # return np.reshape(res, (len(res) / 50, 50, self.config.f_num))

    def _load_YS(self, inputs):
        res = []
        for i in inputs:
            count = 0
            with open(i, 'r') as file:
                for r in file:
                    if count % 50 == 0:
                        res.append(self.s_labels_index.index(int(r.strip())))
                    count += 1
        return one_hot(res, self.config.s_labels_num)

    def _load_YC(self, inputs):
        res = []
        for i in inputs:
            count = 0
            with open(i, 'r') as file:
                for r in file:
                    if count % 750 == 0:
                        res.append(int(r.strip())-1)
                    count += 1
        return one_hot(res, self.config.c_labels_num)

    def get_train_test(self, test_one):
        print("Dataset is now located at: " + self.dataset_path)
        X_INPUT_PATHS = [self.dataset_path + self.X_file_name.format(i + 1) for i in range(self.DAYS)]
        YS_INPUT_PATHS = [self.dataset_path + self.YS_file_name.format(i + 1) for i in range(self.DAYS)]
        YC_INPUT_PATHS = [self.dataset_path + self.YC_file_name.format(i + 1) for i in range(self.DAYS)]
        x_test = self._load_X([X_INPUT_PATHS[test_one]])
        ys_test = self._load_YS([YS_INPUT_PATHS[test_one]])
        ys_test = np.reshape(ys_test, (len(ys_test) / self.config.c_win_size, self.config.c_win_size,
                                       self.config.s_labels_num))
        yc_test = self._load_YC([YC_INPUT_PATHS[test_one]])
        del (X_INPUT_PATHS[test_one])
        del (YS_INPUT_PATHS[test_one])
        del (YC_INPUT_PATHS[test_one])
        x_train = self._load_X(X_INPUT_PATHS)
        ys_train = self._load_YS(YS_INPUT_PATHS)
        ys_train = np.reshape(ys_train, (len(ys_train) / self.config.c_win_size, self.config.c_win_size,
                                       self.config.s_labels_num))
        yc_train = self._load_YC(YC_INPUT_PATHS)

        return x_train, ys_train, yc_train, x_test, ys_test, yc_test

class HuynhOriginalDataset(HuynhDataset):

    X_file_name = 'day{}-data.txt'
    YS_file_name = 'day{}-activities.txt'
    YC_file_name = 'day{}-routines.txt'
    DAYS = 7

    def word_count(self, labels_num, docs, vocabulary):
        vectorizer = CountVectorizer(max_df=1.0, min_df=0, vocabulary=vocabulary,
                                     max_features=labels_num, tokenizer=word_tokenize,
                                     )
        res = vectorizer.fit_transform(docs)
        print ("vocabulary: \nsize:{}\ncontents:{}".format(len(vectorizer.vocabulary_), vectorizer.vocabulary_))
        return res

    def _load_X(self, inputs):
        res = []
        win = self.config.c_win_size * self.config.s_win_size
        for i in inputs:
            with open(i, 'r') as file:
                lines = file.readlines()
                n = len(lines)
                bottom_n = n/win * win
                # del lines
                for r in lines[:bottom_n]:
                    res.append(r.strip().split(' ')[:-2])
        return np.reshape(res, (len(res)/750, 750, self.config.f_num))

    def _load_YS(self, inputs):
        """
        original no data labels: '2', '36'
        no data labels of original data without routines 'unlabels' :
        '2', '4', '5', '6', '9', '12', '25', '28', '29', '31', '32', '36'
        """
        # ys_voc = {}
        # for i in range(0, self.config.s_labels_num):
        #     ys_voc[str(i)] = i
        ys_voc={'0':0, '1': 1, '3':2, '7':3, '8':4, '10':5, '11':6, '13':7, '14':8,
                '15':9, '16':10, '17':11, '18':12, '19':13, '20':14, '21':15,
                '22':16, '23':17, '24':18,'26':19, '27':20, '30':21, '33':22, '34':23, '35':24
        }

        text = []
        win = self.config.c_win_size*self.config.s_win_size
        for i in inputs:
            count = 0
            with open(i, 'r') as file:
                lines = file.readlines()
                for a, b in zip(range(0, len(lines), win),
                                range(win, len(lines) + 1, win)):
                    l = lines[a:b]
                    for c, d in zip(range(0, win, self.config.s_win_size),
                                    range(self.config.s_win_size, win+1, self.config.s_win_size)):
                        text.append(' '.join(l[c:d]).replace('\n', ''))
        r_count = self.word_count(self.config.s_labels_num, text, ys_voc)
        mtx = r_count.todense()
        res = np.argmax(mtx, axis=1)
        res = np.squeeze(np.asarray(res))

        return one_hot(res, self.config.s_labels_num)

    def _load_YC(self, inputs):
        yc_voc = {'1':0, '2': 1, '3':2, '4':3}
        # for i in range(0, self.config.c_labels_num):
        #     yc_voc[str(i)] = i
        win = self.config.c_win_size*self.config.s_win_size
        text = []
        for i in inputs:
            with open(i, 'r') as f:
                lines = f.readlines()
                for a, b in zip(range(0, len(lines), win),
                                range(win, len(lines) + 1, win)):
                    text.append(' '.join(lines[a:b]).replace('\n', ''))

        r_count = self.word_count(self.config.c_labels_num, text, yc_voc)
        mtx = r_count.todense()
        res = np.argmax(mtx, axis=1)
        res = np.squeeze(np.asarray(res))
        return one_hot(res, self.config.c_labels_num)

class OpportunityDataset(DataSet):

    subjects = 4
    runs = 5
    # file_name = "S{}-ADL{}_new.txt"
    file_name = "S{}-ADL{}_valid.txt"

    def word_count(self, labels_num, docs, vocabulary):
        vectorizer = CountVectorizer(max_df=1.0, min_df=0, vocabulary=vocabulary,
                                     max_features=labels_num, tokenizer=word_tokenize,
                                     )
        res = vectorizer.fit_transform(docs)
        print ("vocabulary: \nsize:{}\ncontents:{}".format(len(vectorizer.vocabulary_), vectorizer.vocabulary_))
        return res

    def _crs_to_array(self, crs):
        mtx = crs.todense()
        res = np.argmax(mtx, axis=1)
        res = np.squeeze(np.asarray(res))
        return res

    def _load_X(self, inputs):
        res = []
        win = self.config.c_win_size * self.config.s_win_size

        for i in range(len(inputs)):
            for j in range(len(inputs[0])):
                with open(inputs[i][j], 'r') as file:
                    lines = file.readlines()
                    n = len(lines)
                    bottom_n = n / win * win
                    # print(bottom_n)
                    for r in lines[:bottom_n]:
                        res.append(r.strip().split(' ')[1:-7])
        res = np.array(res, dtype=np.float32)
        return np.reshape(res, (len(res) / win, win, self.config.f_num))

    def _load_YSC(self, inputs):
        yc_voc = {'101':0, '102':1, '103':2, '104':3, '105':4}
        # ys_voc = {'406516':0, '406517':1, '404516':2, '404517':3,
        #         '406520':4, '404520':5, '406505':6, '404505':7, '406519':8, '404519':9,
        #           '406511':10, '404511':11, '406508':12, '404508':13, '408512':14, '407521':15, '405506':16}
        ys_voc = {'1':0, '2':1, '4':2, '5':3}
        text_c = []
        text_s = []
        win = self.config.c_win_size*self.config.s_win_size
        for s in range(len(inputs)):
            for r in range(len(inputs[0])):
                with open(inputs[s][r], 'r') as file:
                    lines = file.readlines()
                    # print(len(lines))
                    for a, b in zip(range(0, len(lines), win),
                                    range(win, len(lines) + 1, win)):
                        l = lines[a:b]
                        lc = [i.split(' ')[-6] for i in l]
                        text_c.append(' '.join(lc))
                        for c, d in zip(range(0, win, self.config.s_win_size),
                                        range(self.config.s_win_size, win+1, self.config.s_win_size)):
                            ls = [j.split(' ')[-7].replace('\r', '').replace('\n', '') for j in l[c:d]]
                            text_s.append(' '.join(ls))
        r_count_s = self.word_count(self.config.s_labels_num, text_s, ys_voc)
        r_count_c = self.word_count(self.config.c_labels_num, text_c, yc_voc)
        # import pdb;pdb.set_trace()
        res_s = self._crs_to_array(r_count_s)
        res_c = self._crs_to_array(r_count_c)

        return one_hot(res_s, self.config.s_labels_num), one_hot(res_c, self.config.c_labels_num)

    def get_train_test(self, test_one):
        print("Dataset is now located at: " + self.dataset_path)
        files = [[] for _ in range(self.subjects)]
        for i in range(self.subjects):
            for j in range(self.runs):
                files[i].append(self.dataset_path + self.file_name.format(i+1, j+1))
        x_test = self._load_X([files[test_one]])
        ys_test, yc_test = self._load_YSC([files[test_one]])
        # import pdb;pdb.set_trace()
        # ys_test = self._load_YS([YS_INPUT_PATHS[test_one]])
        ys_test = np.reshape(ys_test, (len(ys_test) / self.config.c_win_size, self.config.c_win_size,
                                       self.config.s_labels_num))
        # yc_test = self._load_YC([YC_INPUT_PATHS[test_one]])
        del (files[test_one])
        x_train = self._load_X(files)
        ys_train, yc_train = self._load_YSC(files)
        ys_train = np.reshape(ys_train, (len(ys_train) / self.config.c_win_size, self.config.c_win_size,
                                       self.config.s_labels_num))
        # yc_train = self._load_YC(YC_INPUT_PATHS)

        return x_train, ys_train, yc_train, x_test, ys_test, yc_test

class Opportunity113Dataset(DataSet):



    def get_train_test(self, test_one):
        print("Dataset is now located at: " + self.dataset_path)
        f = file(self.dataset_path, 'rb')
        data = cp.load(f)
        f.close()

        X_train, ys_train, yc_train = data[0]
        X_test, ys_test, yc_test = data[1]

        print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        win = self.config.c_win_size * self.config.s_win_size
        c_overlap_size = int(win*self.config.overlap_ratio)
        s_overlap_size = int(self.config.s_win_size * self.config.overlap_ratio)
        # n_train = X_train.shape[0] / win * win
        # n_test = X_test.shape[0] / win * win

        x_train = utils.sliding_window(X_train, (win, self.config.f_num), (c_overlap_size, 1))
        xs_train = []
        for i in x_train:
            new_i = utils.sliding_window(i, (self.config.s_win_size, self.config.f_num), (s_overlap_size, 1))
            xs_train.append(new_i)
        x_train = np.array(xs_train)

        ss_train = utils.sliding_window(ys_train, win, c_overlap_size)
        yss_train = []
        for i in ss_train:
            new_i = utils.sliding_window(i, self.config.s_win_size, s_overlap_size)
            new_i = [j[-1] for j in new_i]
            yss_train.append(new_i)
        ys_train = np.asarray(yss_train)
        ys_train = one_hot(ys_train, self.config.s_labels_num)

        yc_train = np.asarray([i[-1] for i in utils.sliding_window(yc_train, win, c_overlap_size)])
        yc_train = one_hot(yc_train, self.config.c_labels_num)

        x_test = utils.sliding_window(X_test, (win, self.config.f_num), (c_overlap_size, 1))
        xs_test = []
        for i in x_test:
            new_i = utils.sliding_window(i, (self.config.s_win_size, self.config.f_num), (s_overlap_size, 1))
            xs_test.append(new_i)
        x_test = np.array(xs_test)

        ss_test = utils.sliding_window(ys_test, win, c_overlap_size)
        yss_test = []
        for i in ss_test:
            new_i = utils.sliding_window(i, self.config.s_win_size, s_overlap_size)
            new_i = [j[-1] for j in new_i]
            yss_test.append(new_i)
        ys_test = np.asarray(yss_test)
        ys_test = one_hot(ys_test, self.config.s_labels_num)

        yc_test = np.asarray([i[-1] for i in utils.sliding_window(yc_test, win, c_overlap_size)])
        yc_test = one_hot(yc_test, self.config.c_labels_num)

        return x_train, ys_train, yc_train, x_test, ys_test, yc_test
class PelabDataset(DataSet):

    USERS = 4
    Features = [
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
        "SW_X",
        "SW_Y",
        "SW_Z",
        # "BHACC_X",
        # "BHACC_Y",
        # "BHACC_Z",
        # "Breath",
        "HR",
        "BR",
        # "ECG",
    ]

    YS_file_name = 'simple_labels.txt'
    YC_file_name = 'complex_labels.txt'

    def _load_X(self, inputs):
        X_signals = []
        for j in range(len(inputs[0])):
            t = []
            for i in range(len(inputs)):
                # print("access path :{}".format(inputs[i][j]))
                file = open(inputs[i][j], 'r')
                for row in file:
                    t.append(row.strip().split(',')[:-1])
                file.close()
            X_signals.append([np.array(serie, dtype=np.float32) for serie in t])

        # (lines, simple_window, features_num)
        res = np.transpose(np.array(X_signals), (1, 2, 0))
        return np.reshape(res, (res.shape[0] / self.config.c_win_size,
                                      self.config.c_win_size*self.config.s_win_size, self.config.f_num))

    def _load_YS(self, inputs):
        """load simple activity labels"""
        res = []
        for i in inputs:
            # print("access path: {}".format(i))
            with open(i, 'r') as file:
                for r in file:
                    res.append(int(r.strip()) - 1)
        return one_hot(res, self.config.s_labels_num)

    def _load_YC(self, inputs):
        """load complex activity labels"""
        res = []
        for i in inputs:
            # print("access path: {}".format(i))
            count = 0
            with open(i, 'r') as file:
                for r in file:
                    if count % self.config.c_win_size == 0:
                        res.append(int(r.strip()) - 1)
                    count += 1
        return one_hot(res, self.config.c_labels_num)

    def get_train_test(self, test_one):
        print("Dataset is now located at: " + self.dataset_path)
        X_INPUT_PATHS = [[] for _ in range(self.USERS)]
        for i in range(self.USERS):
            for j in range(self.config.f_num):
                X_INPUT_PATHS[i].append(
                    self.dataset_path + "user{}/".format(i + 1) + self.Features[j] + ".txt")  # users * features
        YS_INPUT_PATHS = [self.dataset_path + "user{}/".format(i + 1) + self.YS_file_name for i in range(self.USERS)]
        YC_INPUT_PATHS = [self.dataset_path + "user{}/".format(i + 1) + self.YC_file_name for i in range(self.USERS)]

        x_test = self._load_X([X_INPUT_PATHS[test_one]])
        ys_test = self._load_YS([YS_INPUT_PATHS[test_one]])
        # import pdb;pdb.set_trace()
        ys_test = np.reshape(ys_test, (len(ys_test) / self.config.c_win_size,
                                       self.config.c_win_size, self.config.s_labels_num))
        yc_test = self._load_YC([YC_INPUT_PATHS[test_one]])
        del (X_INPUT_PATHS[test_one])
        del (YS_INPUT_PATHS[test_one])
        del (YC_INPUT_PATHS[test_one])
        x_train = self._load_X(X_INPUT_PATHS)
        ys_train = self._load_YS(YS_INPUT_PATHS)
        ys_train = np.reshape(ys_train, (len(ys_train) / self.config.c_win_size,
                                         self.config.c_win_size, self.config.s_labels_num))
        yc_train = self._load_YC(YC_INPUT_PATHS)

        return x_train, ys_train, yc_train, x_test, ys_test, yc_test


def split_train_test(x, ys, yc):
    chunk_num = 20
    test_sample = [0, 3, 4, 8, 16, 17, 18, 22, 26, 29, 30]
    x_chunks = np.array_split(x, 20, axis=0)
    ys_chunks = np.array_split(np.reshape(ys, (658, 15, 23)), 20, axis=0)
    yc_chunks = np.array_split(yc, 20, axis=0)
    x_train = x_chunks[1]
    ys_train = ys_chunks[1]
    yc_train = yc_chunks[1]
    x_test = x_chunks[0]
    ys_test = ys_chunks[0]
    yc_test = yc_chunks[0]
    count = 2
    for a, b, c in zip(x_chunks[2:], ys_chunks[2:], yc_chunks[2:]):
        if count in test_sample:
            x_test = np.concatenate((x_test, a), axis=0)
            ys_test = np.concatenate((ys_test, b), axis=0)
            yc_test = np.concatenate((yc_test, c), axis=0)
        else:
            x_train = np.concatenate((x_train, a), axis=0)
            ys_train = np.concatenate((ys_train, b), axis=0)
            yc_train = np.concatenate((yc_train, c), axis=0)
        count += 1
    return x_train, ys_train, yc_train, x_test, ys_test, yc_test



# if __name__ == '__main__':
    # x = load_X(X_INPUT_PATHS)
    # print(x.shape)
    # ys = load_YS(YS_INPUT_PATHS)
    # # print(len(ys))
    # print("simple activity shape: {}".format(ys.shape))
    # yc = load_YC(YC_INPUT_PATHS)
    # print(len(yc))
    # print("complex activity shape: {}".format(yc.shape))
    #
    # xt, yst, yct, xet, yset, ycet = split_train_test(x, ys, yc)
    # print("x train: {}".format(xt.shape))
    # print("ys train: {}".format(yst.shape))
    # print("yc train: {}".format(yct.shape))
    # print("x test: {}".format(xet.shape))
    # print("ys test: {}".format(yset.shape))
    # print("yc test: {}".format(ycet.shape))