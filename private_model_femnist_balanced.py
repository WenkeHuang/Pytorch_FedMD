from  data_utils import  get_public_dataset, get_private_dataset_balanced,FEMNIST_iid
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import matplotlib.pyplot as plt
from utils import get_model_list
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class
    """
    def __init__(self,dataset,idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image,label = self.dataset[self.idxs[item]]
        return torch.tensor(image),torch.tensor(label)

def private_dataset_train(args):
    device ='cuda' if args.gpu else 'cpu'
    # 用于初始化模型的部分
    # 获得FEMNIST数据集！
    train_dataset,test_dataset = get_private_dataset_balanced(args)
    user_groups = FEMNIST_iid(train_dataset, args.user_number)

    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN","3_layer_CNN"]

    if args.new_private_training:
        model_list,model_type_list = get_model_list(args.initialurl,modelsindex,models)
    else:
        model_list,model_type_list = get_model_list(args.privateurl,modelsindex,models)


    private_model_private_dataset_train_losses = []
    for n, model in enumerate(model_list):
        print('train Local Model {} on Private Dataset'.format(n))
        model.to(device)
        model.train()
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
        trainloader = DataLoader(DatasetSplit(train_dataset,list(user_groups[n])),batch_size=5,shuffle=True)
        criterion = nn.NLLLoss().to(device)
        train_epoch_losses = []
        print('Begin Private Training')
        for epoch in range(args.privateepoch):
            train_batch_losses = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images,labels = images.to(device),labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 5 ==0:
                    print('Local Model {} Type {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        n,model_type_list[n],epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                train_batch_losses.append(loss.item())
            loss_avg = sum(train_batch_losses)/len(train_batch_losses)
            train_epoch_losses.append(loss_avg)
        torch.save(model.state_dict(),'Src/PrivateModel/LocalModel{}Type{}.pkl'.format(n,model_type_list[n],args.privateepoch))
        # torch.save(model.state_dict(),'Src/PrivateModel/LocalModel{}Type{}Epoch{}.pkl'.format(n,model_type_list[n],args.privateepoch))
        private_model_private_dataset_train_losses.append(train_epoch_losses)
    plt.figure()
    for i,val in enumerate(private_model_private_dataset_train_losses):
        print(val)
        plt.plot(range(len(val)),val,label='model :'+str(i))
    plt.legend(loc='best')
    plt.title('private_model_private_dataset_train_losses')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')
    x_major_locator = MultipleLocator(1)# 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()# ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)# 把x轴的主刻度设置为1的倍数
    plt.xlim(0, args.privateepoch)
    plt.savefig('Src/Figure/private_model_private_dataset_train_losses.png')
    plt.show()
    print('End Private Training')


from option import args_parser
if __name__ == '__main__':
    args = args_parser()
    private_dataset_train(args)




