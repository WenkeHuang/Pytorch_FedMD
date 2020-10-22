from torch import nn
import  torch.nn.functional as F
import copy
class CNN_3layer_fc_model(nn.Module):
    def __init__(self):
        super(CNN_3layer_fc_model, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=128,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.ZeroPad2d(padding=(1,0,1,0)),
                                  nn.AvgPool2d(kernel_size=2, stride=1))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=128,stride=2,
                                            kernel_size=2, out_channels=192),
                                  nn.BatchNorm2d(192),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.AvgPool2d(kernel_size=2))
        self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=192,stride=2,
                                            kernel_size=3, out_channels=256),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2))
        self.FC1 = nn.Linear(2304,16)

    def forward(self, x):
        x = self.CNN1(x)
        # print(x.shape)
        x = self.CNN2(x)
        # print(x.shape)
        x = self.CNN3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        x = self.FC1(x)
        return F.log_softmax(x,dim=1)

class CNN_2layer_fc_model(nn.Module):
    def __init__(self):
        super(CNN_2layer_fc_model, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=128,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.AvgPool2d(kernel_size=2, stride=1))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=128,stride=2,
                                            kernel_size=3, out_channels=256),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2))
        self.FC1 = nn.Linear(43264,16)

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        # (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
        # self.FC = nn.Linear(in_f, 10)  # 全链接层
        x = self.FC1(x)
        return F.log_softmax(x,dim=1)



# def remove_last_layer(model):
#     new_model = copy.deepcopy(model)
#     return new_model


# 在实际训练过程中必须先进行一次前向传播。
# 否则后向传播可能不会更新FC的参数，
# （我的猜测，具体会不会更新我没有试，有兴趣的可以试一下，之后告诉我一下）。
import torch as tc
# import torch
# # 测试CNN_2layer_fc_model与CNN_3lyaer_fc_model结构
# #net = CNN_2layer_fc_model()
# # data_input = tc.autograd.Variable(torch.randn([1, 1, 28, 28]))  # 这里假设输入图片是28*28
# # net = CNN_3layer_fc_model()
# # #print(net(data_input).shape)
# # from torchsummary import summary
# # summary(net,[(1,28,28)])
import torch
class CNN_3layer_fc_model_removelogsoftmax(nn.Module):
    def __init__(self):
        super(CNN_3layer_fc_model_removelogsoftmax, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=128,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.ZeroPad2d(padding=(1,0,1,0)),
                                  nn.AvgPool2d(kernel_size=2, stride=1))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=128,stride=2,
                                            kernel_size=2, out_channels=192),
                                  nn.BatchNorm2d(192),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.AvgPool2d(kernel_size=2))
        self.CNN3 = nn.Sequential(nn.Conv2d(in_channels=192,stride=2,
                                            kernel_size=3, out_channels=256),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2))
        self.FC1 = nn.Linear(2304,16)

    def forward(self, x):
        x = self.CNN1(x)
        # print(x.shape)
        x = self.CNN2(x)
        # print(x.shape)
        x = self.CNN3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        x = self.FC1(x)
        return x

class CNN_2layer_fc_model_removelogsoftmax(nn.Module):
    def __init__(self):
        super(CNN_2layer_fc_model_removelogsoftmax, self).__init__()
        self.CNN1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            kernel_size=3, out_channels=128,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2),
                                  nn.AvgPool2d(kernel_size=2, stride=1))
        self.CNN2 = nn.Sequential(nn.Conv2d(in_channels=128,stride=2,
                                            kernel_size=3, out_channels=256),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.2))
        self.FC1 = nn.Linear(43264,16)

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # 展开
        # print(x.shape)
        # (b, in_f) = x.shape  # 查看卷积层输出的tensor平铺后的形状
        # self.FC = nn.Linear(in_f, 10)  # 全链接层
        x = self.FC1(x)
        return x

# net = CNN_3layer_fc_model_removelogsoftmax()
# net.load_state_dict(torch.load('Src/Model/LocalModel4Type1Epoch10.pkl'))
# net1 = CNN_3layer_fc_model()
# net1.load_state_dict(torch.load('Src/Model/LocalModel4Type1Epoch10.pkl'))
# data_input = torch.autograd.Variable(torch.randn([1, 1, 28, 28]))  # 这里假设输入图片是28*28
# print(net(data_input).sum())
# print(net1(data_input).sum())

