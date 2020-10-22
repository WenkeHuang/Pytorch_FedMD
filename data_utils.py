from torchvision import datasets,transforms
import os
from skimage import  io
import torchvision.datasets.mnist as mnist
from PIL import Image
from torch.utils.data import Dataset,DataLoader

def get_public_dataset(args):
    if args.dataset =='mnist':
        data_dir ='Data/mnist/'
    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root=data_dir,train=True,download=False,transform=apply_transform)
    data_test = datasets.MNIST(root=data_dir,train=False,download=False,transform=apply_transform)
    return data_train,data_test

def convert_to_img(root,train_set,test_set,train=True):
    if (train):
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.item()) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.item()) + '\n')
        f.close()

def init_private_dataset(args):
    if args.private_dataset == 'FEMNIST':
        rootraw = "Data/FEMNIST/raw/"
        root = "Data/FEMNIST/"
    train_set = (
        mnist.read_image_file(os.path.join(rootraw, 'emnist-letters-train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(rootraw, 'emnist-letters-train-labels-idx1-ubyte'))
    )

    test_set = (
        mnist.read_image_file(os.path.join(rootraw, 'emnist-letters-test-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(rootraw, 'emnist-letters-test-labels-idx1-ubyte'))
    )

    print("train set:", train_set[0].size())
    print("test set:", test_set[0].size())

    convert_to_img(root,train_set,test_set,train=True)
    convert_to_img(root,train_set,test_set,train=False)


def default_loader(path):
    return Image.open(path)

# 用于获得未处理过的私有数据集
def get_private_dataset(args):
    if args.private_dataset == 'FEMNIST':
        root = "Data/FEMNIST/"
    class MyDataset(Dataset):
        # 当我们对类的属性item进行下标的操作时，首先会被__getitem__()、__setitem__()、__delitem__()拦截，从而执行我们在方法中设定的操作，如赋值，修改内容，删除内容等等。
        def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
            fh = open(txt, 'r')
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1]))) # 在这里对数据进行修改
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)
    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    data_train = MyDataset(txt=root + 'train.txt', transform=apply_transform)
    data_test = MyDataset(txt=root + 'test.txt', transform=apply_transform)
    return  data_train,data_test

# 用于获得处理过的数据集限制了标签的值
def get_private_dataset_balanced(args):
    if args.private_dataset == 'FEMNIST':
        root = "Data/FEMNIST/"
    private_data_index = [10,11,12,13,14,15]
    class MyDataset(Dataset):
        # 当我们对类的属性item进行下标的操作时，首先会被__getitem__()、__setitem__()、__delitem__()拦截，从而执行我们在方法中设定的操作，如赋值，修改内容，删除内容等等。
        def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
            fh = open(txt, 'r')
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                if (int(words[1])+9) in private_data_index:
                    imgs.append((words[0], (int(words[1])+9))) # 在这里对数据进行修改
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    data_train = MyDataset(txt=root + 'train.txt', transform=apply_transform)
    data_test = MyDataset(txt=root + 'test.txt', transform=apply_transform)
    return  data_train,data_test

import  numpy as np

def FEMNIST_iid(dataset,num_users):
    num_item = int(len(dataset)/num_users)
    num_item = 20 # 用于达到作者目的！
    dict_users, all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,num_item,replace=False))
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users



class args:
    dataset = 'mnist'
    private_dataset = 'FEMNIST'

# get_public_dataset(args)
# init_private_dataset(args)
# # data_train,data_test = get_private_dataset(args)
# data_train,data_test = get_private_dataset_balanced(args)
# print(FEMNIST_iid(data_train,10)[9])