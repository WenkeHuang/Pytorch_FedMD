# import matplotlib.pyplot as plt
# demo1 = [1,1,2,3,4]
# demo2 = [1,1,5,3,4]
# demo = []
# demo.append(demo1)
# demo.append(demo2)
#
# plt.figure()
# for i,val in enumerate(demo):
#     print(val)
#     plt.plot(range(len(val)),val)
#     print(i)
# plt.savefig('Src/Figure/Demo.png')
# plt.show()
#
# import  torch
# print(torch.cuda.current_stream())
# import random
# tempindex = random.randint(0, 1)
# for i in range(10):
#     print(random.randint(0,1))
# from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
# model1 = CNN_2layer_fc_model()
# model2 = CNN_2layer_fc_model()





# import os
# from skimage import io
# import torchvision.datasets.mnist as mnist
# import numpy
#
# rootraw = "Data/FEMNIST/raw/"
# root = "Data/FEMNIST/"
# train_set = (
#     mnist.read_image_file(os.path.join(rootraw, 'emnist-letters-train-images-idx3-ubyte')),
#     mnist.read_label_file(os.path.join(rootraw, 'emnist-letters-train-labels-idx1-ubyte'))
# )
#
# test_set = (
#     mnist.read_image_file(os.path.join(rootraw, 'emnist-letters-test-images-idx3-ubyte')),
#     mnist.read_label_file(os.path.join(rootraw, 'emnist-letters-test-labels-idx1-ubyte'))
# )
#
# print("train set:", train_set[0].size())
# print("test set:", test_set[0].size())
#
#
# def convert_to_img(train=True):
#     if (train):
#         f = open(root + 'train.txt', 'w')
#         data_path = root + '/train/'
#         if (not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
#             img_path = data_path + str(i) + '.jpg'
#             io.imsave(img_path, img.numpy())
#             f.write(img_path + ' ' + str(label.item()) + '\n')
#         f.close()
#     else:
#         f = open(root + 'test.txt', 'w')
#         data_path = root + '/test/'
#         if (not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
#             img_path = data_path + str(i) + '.jpg'
#             io.imsave(img_path, img.numpy())
#             f.write(img_path + ' ' + str(label.item()) + '\n')
#         f.close()
#
#
# convert_to_img(True)
# convert_to_img(False)
#
# import torch
# from torch.autograd import Variable
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# root = "Data/FEMNIST/"
#
# # -----------------ready the dataset--------------------------
# def default_loader(path):
#     return Image.open(path).convert('RGB')
# class MyDataset(Dataset):
#     def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
#         fh = open(txt, 'r')
#         imgs = []
#         for line in fh:
#             line = line.strip('\n')
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0],int(words[1])))
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = self.loader(fn)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img,label
#
#     def __len__(self):
#         return len(self.imgs)
#
# train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
# test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_data, batch_size=64)

import matplotlib.pyplot as plt
# import torch
# from data_utils import get_private_dataset_balanced
# class args:
#     private_dataset = 'FEMNIST'
#
# data_train,data_test = get_private_dataset_balanced(args)
# data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=True)  #600*100*([[28*28],x])
#
# for i, (images, labels) in enumerate(data_loader_train):
#
#     print(labels)
#     # '''
#     #     每一个周期，共600个批次（i=0~599）；
#     #     data_loader_train包含600个批次，包括整个训练集；
#     #     每一批次一共100张图片，对应100个标签, len(images[0])=1；
#     #     images包含一个批次的100张图片（image[0].shape=torch.Size([1,28,28])），labels包含一个批次的100个标签，标签范围为0~9
#     # '''
#     #
#     # #每100个批量绘制绘制最后一个批量的所有图片
#     # if (i ) % 10 == 0:
#     #     print('batch_number [{}/{}]'.format(i + 1, len(data_train)))
#     #     for j in range(len(images)):
#     #         image = images[j].resize(28, 28) #将(1,28,28)->(28,28)
#     #         plt.imshow(image)  # 显示图片,接受tensors, numpy arrays, numbers, dicts or lists
#     #         plt.axis('off')  # 不显示坐标轴
#     #         plt.title("$The {} picture in {} batch, label={}$".format(j + 1, i + 1, labels[j]))
#     #         plt.show()
# import torch
# from torch import nn
# #
# # loss = nn.NLLLoss(reduction="none")
# # input = torch.tensor(([1.,2.,3.],[4.,5.,6.]))
# # target = torch.tensor([5,4])
# # output = loss(input, target)
# # print(output)
#
# '''
# 输出：tensor([[2., 0., 2.],
#               [4., 0., 4.]])
# '''
# import  numpy as np
# list = []
# list1 = [1,2,3]
# list2 = [2,3,4]
# list.append(list1)
# list.append(list2)
# print(np.array(list).mean(axis=0))

#
from option import args_parser
if __name__ == '__main__':
    args = args_parser()
    print(args.new_collaborative_training)
    args.new_collaborative_training = True
    print(args.new_collaborative_training)

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys

# 列出目录
print ("目录为: %s"%os.listdir('Src/CollaborativeModel'))

for item in os.listdir('Src/CollaborativeModel'):
    os.rename(item)
# 重命名
# os.rename("test","test2")

# print "重命名成功。"

# 列出重命名后的目录
