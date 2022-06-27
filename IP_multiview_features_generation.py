import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import cv2
import h5py
from sklearn import preprocessing


def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index - PATCH_SIZE, height_index + PATCH_SIZE)
    width_slice = slice(width_index - PATCH_SIZE, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch


def generate_ft_test_data_new(seed_number):
    img = sio.loadmat('D:\hyperspectral_data\Indian_pines.mat')
    img = img['indian_pines_corrected']
    m, n, b = img.shape

    gt = sio.loadmat('D:\hyperspectral_data\Indian_pines_gt.mat')
    gt = gt['indian_pines_gt'].reshape(-1)
    print("gt, gt.shape,  np.unique(gt)", gt, gt.shape, np.unique(gt))
    # [0 0 0 ... 0 0 0] (207400,) [0 1 2 3 4 5 6 7 8 9]

    gt_ = []
    for each in gt:
        if each != 0:
            gt_.append(each)
    gt_ = np.array(gt_)
    print("gt_, gt_.shape, np.unique(gt_)", gt_, gt_.shape, np.unique(gt_))
    # [1 1 1 ... 2 2 2] (42776,) [1 2 3 4 5 6 7 8 9]

    # -----------------------------------------------------------------#

    PATCH_SIZE = 16

    # cv2.BORDER_REFLECT-边界元素的镜像方式
    img = cv2.copyMakeBorder(img, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_REFLECT)
    gt = gt.reshape(-1)
    # gt = cv2.copyMakeBorder(gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

    img = (img - img.min()) / (img.max() - img.min())

    indices = np.arange(b)
    random_index = np.random.permutation(indices)[:30]

    img = img[:, :, random_index]
    [mm, nn, bb] = img.shape

    train_dataset = []

    for i in range(PATCH_SIZE, mm - PATCH_SIZE):
        for j in range(PATCH_SIZE, nn - PATCH_SIZE):
            gt_index = (i - PATCH_SIZE) * n + j - PATCH_SIZE
            if gt[gt_index] != 0:
                temp = Patch(img, i, j, PATCH_SIZE).reshape(PATCH_SIZE * 2, PATCH_SIZE * 2, bb)
                train_dataset.append(temp)  # 20, 28, 28, 3

    train_dataset = np.array(train_dataset)
    print("train_dataset.shape", train_dataset.shape)
    # (42776, 28, 28, 30)

    test_dataset = train_dataset
    test_dataset = test_dataset.reshape(-1, PATCH_SIZE*2, PATCH_SIZE*2, 3, 10).transpose(4, 0, 1, 2, 3)
    print("test_dataset.shape: ", test_dataset.shape)  # (10, 42776, 28, 28, 3)
    print("gt_.shape: ", gt_.shape)  # (42776,)

    f = h5py.File(
        './data/IP_' + str(test_dataset.shape[0]) + '_' + str(test_dataset.shape[1]) + '_' + str(test_dataset.shape[2])
        + '_' + str(test_dataset.shape[3]) + '_' + str(test_dataset.shape[4]) + '.h5', 'w')
    f['data'] = test_dataset  # (10, 42776, 28, 28, 3)
    f['label'] = gt_ - 1  # (42776,)
    f.close()  # PU_42776_28_28_3

    # -----------------------------------------------------------------#

    np.random.seed(int(seed_number))

    indices = np.arange(train_dataset.shape[0])
    shuffled_indices = np.random.permutation(indices)
    train_dataset = train_dataset[shuffled_indices]
    gt_ = gt_[shuffled_indices]

    data = []
    sample_preclass = 5

    for class_index in range(gt_.max()):  # 类别0-8
        count = 0
        for index in range(train_dataset.shape[0]):  # 数量0-4
            if gt_[index] == class_index + 1 and count < sample_preclass:
                data.append(train_dataset[index])
                count += 1
    data = np.array(data).reshape(-1, PATCH_SIZE*2, PATCH_SIZE*2, 3, 10).transpose(4, 0, 1, 2, 3)

    #num_s = 5
    gt = np.array(range(16))[:, np.newaxis]
    gt = np.repeat(gt, sample_preclass, axis=1).reshape(-1)
    print("data.shape", data.shape)  # (450, 28, 28, 3)
    print("gt", gt.shape)  # (450,)

    f = h5py.File('./data/IP_' + str(data.shape[2]) + '_' + str(data.shape[3]) + '_' + str(data.shape[4]) + '_' + str(
        sample_preclass) + 'perc.h5', 'w')
    f['data'] = data  # (45, 8100)
    f['label'] = gt  # (45, 9)
    f.close()

    # ----------------------------------------------------------------------------#


# generate_ft_data()
import sys
# seed_number = "88"
seed_number = sys.argv[1]


generate_ft_test_data_new(seed_number)