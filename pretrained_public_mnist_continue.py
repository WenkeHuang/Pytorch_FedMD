from  data_utils import  get_public_dataset
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

class args:
    gpu =1
    dataset ='mnist'
    initialurl = 'Src\Model'
    continue_epoch = 10
    lr = 0.01
    optimizer ='sgd'

def get_model_list(url,modelsindex,models):
    model_list = []
    model_type_list = []
    filePath = url
    for root, dirs, files in os.walk(filePath, topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            # print(name[name.find('Type')+4])
            model_type_list.append(int(name[name.find('Type')+4]))
            net = models[modelsindex[int(name[name.find('Type')+4])]]()
            net.load_state_dict(torch.load(os.path.join(root, name)))
            model_list.append(net)
    return model_list,model_type_list

def continue_train_models(args):
    device = 'cuda' if args.gpu else 'cpu'
    # 用于初始化模型的部分
    train_dataset, test_dataset = get_public_dataset(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
              "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN", "3_layer_CNN"]
    model_list, model_type_list = get_model_list(args.initialurl, modelsindex, models)

    private_model_public_dataset_train_losses = []
    for n, model in enumerate(model_list):
        print('Continue train Local Model {}'.format(n))
        model.to(device)
        model.train()
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)
        trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        criterion = nn.NLLLoss().to(device)
        train_epoch_losses = []
        print('Begin Public Training')
        for epoch in range(args.continue_epoch):

            train_batch_losses = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('Local Model {} Type {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        n, model_type_list[n], epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                                               100. * batch_idx / len(trainloader), loss.item()))
                train_batch_losses.append(loss.item())
            loss_avg = sum(train_batch_losses) / len(train_batch_losses)
            train_epoch_losses.append(loss_avg)

        torch.save(model.state_dict(),
                   'Src/Model/LocalModel{}Type{}Epoch{}.pkl'.format(n, model_type_list[n], args.epoch))
        private_model_public_dataset_train_losses.append(train_epoch_losses)

    plt.figure()
    for i, val in enumerate(private_model_public_dataset_train_losses):
        print(val)
        plt.plot(range(len(val)), val)
    plt.xlabel('epoches')
    plt.ylabel('Train loss')
    plt.savefig('Src/Figure/private_model_public_dataset_train_continue_losses.png')
    plt.show()
    print('End Public Training')

from option import args_parser
if __name__ == '__main__':
    args = args_parser()
    continue_train_models(args)






