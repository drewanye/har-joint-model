__author__ = 'zhenanye'


import numpy as np

# -------
# load Huynh dataset
# -------

DATA_PATH = 'data/'
DATASET_PATH = DATA_PATH + 'Huynh Dataset/'

X_file_name = 'day{}-data.txt'
YS_file_name = 'day{}-activity.txt'
YC_file_name = 'day{}-routine.txt'
DAYS = 7
Features_num = 12
s_labels_index = [1, 3, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 30, 33, 34, 35]


X_INPUT_PATHS = [DATASET_PATH + X_file_name.format(i+1) for i in range(DAYS)]
YS_INPUT_PATHS = [DATASET_PATH + YS_file_name.format(i+1) for i in range(DAYS)]
YC_INPUT_PATHS = [DATASET_PATH + YC_file_name.format(i+1) for i in range(DAYS)]

def one_hot(y_, n_values):
    #y_ = y_.reshape(len(y_))
    # n_values = int(np.max(y_))+1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def load_X(inputs):
    res = []
    for i in inputs:
        with open(i, 'r') as file:
            for r in file:
                res.append(r.strip().split(',')[:-1])
    return np.reshape(res, (len(res)/750, 750, Features_num))

def load_YS(inputs):
    res = []
    for i in inputs:
        count = 0
        with open(i, 'r') as file:
            for r in file:
                if count % 50 == 0:
                    res.append(s_labels_index.index(int(r.strip())))
                count += 1
    return one_hot(res, 23)

def load_YC(inputs):
    res = []
    for i in inputs:
        count = 0
        with open(i, 'r') as file:
            for r in file:
                if count % 750 == 0:
                    res.append(int(r.strip())-1)
                count += 1
    return one_hot(res, 4)



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

def get_train_test(test_day, x_inputs, ys_inputs, yc_inputs):
    x_test = load_X([x_inputs[test_day]])
    ys_test = load_YS([ys_inputs[test_day]])
    ys_test = np.reshape(ys_test, (len(ys_test)/15, 15, 23))
    yc_test = load_YC([yc_inputs[test_day]])
    del(x_inputs[test_day])
    del(ys_inputs[test_day])
    del(yc_inputs[test_day])
    x_train = load_X(x_inputs)
    ys_train = load_YS(ys_inputs)
    ys_train = np.reshape(ys_train, (len(ys_train)/15, 15, 23))
    yc_train = load_YC(yc_inputs)

    return x_train, ys_train, yc_train, x_test, ys_test, yc_test

# test
if __name__ == '__main__':
    x = load_X(X_INPUT_PATHS)
    print(x.shape)
    ys = load_YS(YS_INPUT_PATHS)
    # print(len(ys))
    print("simple activity shape: {}".format(ys.shape))
    yc = load_YC(YC_INPUT_PATHS)
    # print(len(yc))
    print("complex activity shape: {}".format(yc.shape))

    xt, yst, yct, xet, yset, ycet = split_train_test(x, ys, yc)
    print("x train: {}".format(xt.shape))
    print("ys train: {}".format(yst.shape))
    print("yc train: {}".format(yct.shape))
    print("x test: {}".format(xet.shape))
    print("ys test: {}".format(yset.shape))
    print("yc test: {}".format(ycet.shape))