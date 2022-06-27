import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import cv2
import h5py
from sklearn import preprocessing

img = h5py.File('Chikusei.mat')['chikusei'][:].transpose(1,2,0)
gt = sio.loadmat('Chikusei_gt.mat')['GT'][0][0][0]
m, n, b = img.shape
print(m, n, b)



# 统计每类样本个数
# num=np.zeros(gt.max()+1)
# for i in range(m):
#     for j in range(n):
#        num[gt[i,j]]=num[gt[i,j]]+1
#
# print(num) #[164624.   6631.  18649.   2099.   3064.   1345.   5029.   1330.   3682. 947.]


def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1, patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch

def generate_PCAfeatures_for_traning(img, gt):

    np.random.seed(123456789)

    # 先分组 再降维
    ##################################
    PATCH_SIZE = 14

    # cv2.BORDER_REFLECT-边界元素的镜像方式
    print(type(img))
    img = cv2.copyMakeBorder(img, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_REFLECT)
    gt = gt.reshape(-1)
    #gt = cv2.copyMakeBorder(gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))


    img = (img - img.min()) / (img.max() - img.min())
    [mm, nn, bb] = img.shape

    data_division = []

    number = 30704

    indices = np.arange(77592)  # list [0,.....,42775]
    shuffled_indices = np.random.permutation(indices)[:number]

    count = 0
    num_group = 20

    for i in range(PATCH_SIZE, mm - PATCH_SIZE):
        for j in range(PATCH_SIZE, nn - PATCH_SIZE):
            gt_index = (i - PATCH_SIZE) * n + j - PATCH_SIZE
            if gt[gt_index] != 0:
                if count in shuffled_indices:
                    print(count)
                    temp = Patch(img, i, j, PATCH_SIZE).reshape(PATCH_SIZE * 2, PATCH_SIZE * 2, bb)
                    data_list = []
                    for num in range(num_group):
                        indices_ = np.arange(bb)
                        shuffled_indices_ = np.random.permutation(indices_)[0:3]
                      #print(temp[:,:,shuffled_indices].shape)
                        data_list.append(temp[:,:,shuffled_indices_]) # 20, 28, 28, 3
                    #print(np.array(data_list).shape)
                    #print(num)

                    data_division.append(data_list) # 42776, 20, 28, 28, 3
                count += 1
            #print(count)

    print(i, j)
    data_division = np.array(data_division)
    print(data_division.shape)


    f = h5py.File(
        './CH_' + str(number) + '_' + str(num_group) + '_' + str(PATCH_SIZE * 2) + '_' + str(PATCH_SIZE * 2) + '_3.h5',
        'w')
    f['data'] = data_division
    gt = np.delete(gt, np.where(gt == 0))-1
    print(np.unique(gt))
    f['label'] = gt
    f.close()

generate_PCAfeatures_for_traning(img, gt)