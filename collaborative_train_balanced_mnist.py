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

def collaborative_private_model_mnist_train(args):

    device ='cuda' if args.gpu else 'cpu'
    # 用于初始化模型的部分
    train_dataset,test_dataset = get_public_dataset(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model}
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

        trainloader = DataLoader(DatasetSplit(train_dataset,list(epoch_groups[epoch])),batch_size=128,shuffle=True)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images,labels = images.to(device),labels.to(device)
            temp_sum_result = []
            # Make output together
            for n, model in enumerate(model_list):
                model.to(device)
                model.eval()
                outputs = model(images)
                _,pred_labels = torch.max(outputs,1)
                pred_labels = pred_labels.view(-1)
                temp_sum_result.append(pred_labels)
            temp_sum_result = torch.stack(temp_sum_result)
            labels = torch.mean(temp_sum_result.float(),dim=0) # get the output
            labels = labels.type(torch.LongTensor)
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
                criterion = nn.NLLLoss().to(device)
                optimizer.zero_grad()
                outputs = model(images)
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
