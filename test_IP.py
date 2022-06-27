import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time
from sklearn import metrics

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 256)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 8)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 16)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
#parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class
#parser.add_argument("-e","--episode",type = int, default= 1)
#-----------------------------------------------------------------------------------#
#parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=3)
parser.add_argument("-i", "--index", type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
#FEATURE_DIM = args.feature_dim
#RELATION_DIM = args.relation_dim
n_way = args.n_way
n_shot = args.n_shot
#n_query = args.n_query
#EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

#n_examples = 200  # 训练数据集中每类200个样本
im_width, im_height, channels = 28, 28, 3 # 输入的cube为固定值

num_fea = 128
num_fea_2 = num_fea*2
num_fea_3 = num_fea_2*2
num_fea_4 = num_fea_3*2

class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))

    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, num_fea, kernel_size=1, padding=0),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU())

        self.res1 = nn.Sequential(
                        nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU(),
                        nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_2),
            nn.ReLU())

        self.res2 = nn.Sequential(
                        nn.Conv2d(num_fea_2, num_fea_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU(),
                        nn.Conv2d(num_fea_2, num_fea_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_fea_2, num_fea_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_3),
            nn.ReLU())

        self.res3 = nn.Sequential(
                        nn.Conv2d(num_fea_3,num_fea_3,kernel_size=3,padding=1),
                        nn.BatchNorm2d(num_fea_3),
                        nn.ReLU(),
                        nn.Conv2d(num_fea_3, num_fea_3, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_3),
                        nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_fea_3, num_fea_4, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_fea_4),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(num_fea_4, num_fea_4, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_fea_4),
            nn.ReLU())

    def forward(self,x):

        out = self.layer1(x)
        out1 = self.res1(out) + out
        out1 = self.maxpool(out1)

        out2 = self.layer2(out1)
        out2 = self.res2(out2) + out2
        out2 = self.maxpool(out2)

        out3 = self.layer3(out2)
        out4 = self.res3(out3) + out3
        out4 = self.maxpool(out4)

        out5 = self.layer4(out4)
        out5 = self.layer5(out5)


        #out = out.view(out.size(0),-1)
        #print(list(out5.size())) # [100, 128, 1, 1]
        return out5 # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(num_fea_4*2, 512, kernel_size=1, padding=0),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p = 0.5)                                                                              # 测试的时候需要修改....？？？

    def forward(self,x): # [7600, 128, 2, 2]
        out = self.layer1(x)
        #print(list(out.size()))
        #print(list(out.size())) # [6000, 128, 2, 2]
        out = out.view(out.size(0),-1) # flatten
        #print(list(out.size())) # [6000, 512]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.sigmoid(self.fc2(out))
        #print("ssss", list(out.size())) # [6000, 1]
        return out


feature_encoder = CNNEncoder()
relation_network = RelationNetwork()

feature_encoder.cuda(GPU)
relation_network.cuda(GPU)

feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)


feature_encoder.load_state_dict(torch.load(str("./model/IP_feature_encoder_16way_2shot_1000FT.pkl")))
print("load feature encoder success")

relation_network.load_state_dict(torch.load(str("./model/IP_relation_network_16way_2shot_1000FT.pkl")))
print("load relation network success")

feature_encoder.eval()
relation_network.eval()


def rn_predict(support_images, test_images, num):

    support_tensor = torch.from_numpy(support_images).type(torch.FloatTensor)
    query_tensor = torch.from_numpy(test_images).type(torch.FloatTensor)

    # calculate features
    sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
    #print("list(sample_features.size())", list(sample_features.size()) ) # [45, 256, 1, 1]
    sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-3])  # view函数改变shape:

    # sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
    sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均

    #print("list(sample_features.size())", list(sample_features.size()) ) # [9, 256]
    batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))
    #print( "list(batch_features.size())", list(batch_features.size())) # [1000, 256, 1, 1]
    batch_features = batch_features.view(list(batch_features.size())[0], list(batch_features.size())[1])
    #print("list(batch_features.size())", list(batch_features.size()))  # [1000, 256]

    # calculate relations
    sample_features_ext = sample_features.unsqueeze(0).repeat(num, 1, 1)  # # repeat函数沿着指定的维度重复tensor
    #print("list(sample_features_ext.size())", list(sample_features_ext.size())) # [9000, 9, 256]
    batch_features_ext = batch_features.unsqueeze(0).repeat(n_way, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    #print("list(batch_features_ext.size())", list(batch_features_ext.size())) # [1000, 9, 256]

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
    #print("list(relation_pairs.size())", list(relation_pairs.size())) # [180, 20, 512]
    relation_pairs = relation_pairs.view(-1, list(relation_pairs.size())[-1], 1, 1)
    #print("list(relation_pairs.size())", list(relation_pairs.size())) # [3600, 512, 1, 1]

    relations = relation_network(relation_pairs)
    #print("list(relations.size())", list(relations.size())) # [3600, 1]
    relations = relations.view(-1, n_way)
    #print("list(relations.size())", list(relations.size())) # [180, 20]

    # 得到预测标签
    _, predict_label = torch.max(relations.data, 1)
    # print('predict_label', predict_label)

    return predict_label


def test():

    A = time.time()
    # 加载支撑数据
    f = h5py.File(r'./data/IP_28_28_3_5perc.h5', 'r')
    support_images = np.array(f['data'][:][args.index])  # (5, 8100)
    support_images = support_images.reshape(-1, 28, 28, 3).transpose((0, 3, 1, 2))
    print('support_images = ', support_images.shape)  # (9, 1, 100, 9, 9)
    f.close()

    # 加载测试
    f = h5py.File(r'./data/IP_10_10249_28_28_3.h5', 'r')  # 路径
    test_images = np.array(f['data'])[args.index]  # (42776, 8100)
    test_images = test_images.reshape(-1, 28, 28, 3).transpose((0, 3, 1, 2))
    print('test_images = ', test_images.shape)  # (42776, 1, 100, 9, 9)
    test_labels = f['label'][:]  # (42776, )
    f.close()

    #epi_classes = np.random.permutation(test_images.shape[0])
    #test_images = test_images[epi_classes, :, :, :, :]
    #test_labels = test_labels[epi_classes]

    predict_labels = []  # 记录预测标签
    # S1
    for i in range(0, 102):#10988 42776
        test_images_ = test_images[100 * i:100 * (i + 1), :, :, :]
        predict_label = rn_predict(support_images, test_images_, num = 100)
        predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S2
    test_images_ = test_images[-49:, :, :, :]
    predict_label = rn_predict(support_images, test_images_, num = 49)
    predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S3
    #print(test_labels.shape) # (42776,)
    print(np.unique(predict_labels))
    #print(np.array(predict_labels).shape) # (42776,)
    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(test_images.shape[0])]

    ##################### 混淆矩阵 #####################
    matrix = metrics.confusion_matrix(test_labels, predict_labels)
    print(matrix)
    OA = np.sum(np.trace(matrix)) / 10249.0 * 100.0
    print('OA = ', round(OA, 2))

    # print(rewards)
    #total_rewards = np.sum(rewards)
    # print(total_rewards)

    #print(time.time()-A)

    #accuracy = total_rewards / test_images.shape[0]
    #print("accuracy:", accuracy)

    # f = open('./PU_prediction_' + str(round(OA, 2)) + '_' + str(args.index) + '.txt', 'w')
    # for i in range(test_images.shape[0]):
    #     f.write(str(predict_labels[i]) + '\n')

    np.save("IP_" + str(args.index) + ".npy", predict_labels)


if __name__ == '__main__':
    test()



"""Pytorch中神经网络模块化接口nn的了解"""
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)

    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.

    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：

    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28

"""