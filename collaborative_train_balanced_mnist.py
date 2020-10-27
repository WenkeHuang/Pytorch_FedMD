from  data_utils import  get_public_dataset,MNIST_random
from  models import CNN_2layer_fc_model_removelogsoftmax,CNN_3layer_fc_model_removelogsoftmax,CNN_3layer_fc_model,CNN_2layer_fc_model
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import matplotlib.pyplot as plt
from utils import get_model_list

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

def list_add(a,b):
    c = []
    for i in range(len(a)):
        d = []
        for j in range(len(a[i])):
            d.append(a[i][j]+b[i][j])
        c.append(d)
    return c

def get_avg_result(temp_sum_result,num_client):
    for itemx in range(len(temp_sum_result)):
        for itemy  in range(len(temp_sum_result[itemx])):
            temp_sum_result[itemx][itemy] /=num_client
    return temp_sum_result

def collaborative_private_model_mnist_train(args):

    device ='cuda' if args.gpu else 'cpu'
    # 用于初始化模型的部分
    train_dataset,test_dataset = get_public_dataset(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model_removelogsoftmax,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model_removelogsoftmax}
    modelsindex = ["2_layer_CNN","3_layer_CNN"]
    if args.new_collaborative_training:
        model_list,model_type_list = get_model_list(args.privateurl,modelsindex,models)
    else:
        model_list,model_type_list = get_model_list(args.Collaborativeurl,modelsindex,models)
    epoch_groups = MNIST_random(train_dataset,args.collaborative_epoch)

    train_loss = []
    test_accuracy = []
    for i in range(args.user_number):
        train_loss.append([])
    for i in range(args.user_number):
        test_accuracy.append([])


    for epoch in range(args.collaborative_epoch):

        train_batch_loss = []
        for i in range(args.user_number):
            train_batch_loss.append([])

        trainloader = DataLoader(DatasetSplit(train_dataset,list(epoch_groups[epoch])),batch_size=256,shuffle=True)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images,labels = images.to(device),labels.to(device)
            # 初始化存储结果的东西
            temp_sum_result = [ [] for _ in range(len(labels))]
            for item in range(len(temp_sum_result)):
                for i in range(args.output_classes):
                    temp_sum_result[item].append(0)

            # Make output together
            for n, model in enumerate(model_list):
                with torch.no_grad():
                    model.to(device)
                    model.eval()
                    outputs = model(images)
                    pred_labels = outputs.tolist() # 转成list
                    # print(pred_labels.shape) # torch.Size([128, 16])
                    # _,pred_labels = torch.max(outputs,1)
                    # pred_labels = pred_labels.view(-1)
                    # print(pred_labels.shape) # torch.Size([2048])
                    temp_sum_result = list_add(pred_labels,temp_sum_result) # 把每次的结果都给加到一起
            #         print(len(temp_sum_result))
            #         print(len(temp_sum_result[0]))
            # print(type(temp_sum_result))
            # print(type(temp_sum_result[0]))
            # temp_sum_result = torch.stack(temp_sum_result) # torch.Size([10, 128, 16])
            # temp_sum_result /=args.user_number
            # print(temp_sum_result.shape)
            # labels = torch.mean(temp_sum_result.float(),dim=0) # get the output
            # print(labels.shape) # torch.Size([128, 16])
            temp_sum_result = get_avg_result(temp_sum_result,args.user_number) # 根据参与训练的时候用户把结果除以对应的数量
            labels = torch.tensor(temp_sum_result)
            # print(labels.size())
            # print((labels[0]).size())
            labels = labels.to(device)
            for n,model in enumerate (model_list):
                model.to(device)
                model.train()
                if args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                                momentum=0.5)
                elif args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                                 weight_decay=1e-4)
                criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean').to(device)
                optimizer.zero_grad()
                outputs = model(images) # torch.Size([128, 16])
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 ==0:
                    print('Collaborative traing : Local Model {} Type {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        n,model_type_list[n],epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                train_batch_loss[n].append(loss.item())
                torch.save(model.state_dict(),'Src/CollaborativeModel/LocalModel{}Type{}.pkl'.format(n,model_type_list[n]))
        for index in range(len(train_batch_loss)):
            loss_avg =  sum(train_batch_loss[index])/len(train_batch_loss[index])
            train_loss[index].append(loss_avg)

    plt.figure()
    for index in range(len(train_loss)):
        plt.plot(range(len(train_loss[index])), train_loss[index])
    plt.title('collaborative_train_losses')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')
    plt.savefig('Src/Figure/collaborative_train_losses.png')
    plt.show()
    print('End Public Training')



from option import args_parser
if __name__ == '__main__':
    args = args_parser()
    collaborative_private_model_mnist_train(args)
