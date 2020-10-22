import  torch
from  data_utils import  get_public_dataset
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import tqdm
import random
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class args:
    gpu =1
    dataset ='mnist'
    user_number = 10
    lr = 0.01
    optimzier ='sgd'
    epoch = 5

def train_models(device,models,modelsindex,train_dataset,test_dataset,lr,optimizer,epochs):
    '''
    Train an array of models on the same dataset.
    We use early termination to speed up training.
    '''
    private_model_public_dataset_train_losses = []
    for n, model in enumerate(models):
        print("Training model ", n)
        print('Mdeol Type index',modelsindex[n])
        model.to(device)
        model.train()
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.5)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # batch_size = 64 打乱 True
        criterion = torch.nn.NLLLoss().to(device)
        train_epoch_losses = []
        print('Begin Public Training')
        for epoch in range(epochs):
            train_batch_losses = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images,labels = images.to(device),labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 ==0:
                    print('Local Model {} Type {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        n,modelsindex[n],epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                train_batch_losses.append(loss.item())
            loss_avg = sum(train_batch_losses)/len(train_batch_losses)
            train_epoch_losses.append(loss_avg)
        torch.save(model.state_dict(),'Src/Model/LocalModel{}Type{}Epoch{}.pkl'.format(n,modelsindex[n],epochs))
        private_model_public_dataset_train_losses.append(train_epoch_losses)
    plt.figure()
    for i,val in enumerate(private_model_public_dataset_train_losses):
        print(val)
        plt.plot(range(len(val)),val)
    plt.xlabel('epoches')
    plt.ylabel('Train loss')
    plt.savefig('Src/Figure/private_model_public_dataset_train_losses.png')
    plt.show()
    print('End Public Training')


if __name__ == '__main__':

    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    device ='cuda' if args.gpu else 'cpu'

    # 用于初始化模型的部分
    train_dataset,test_dataset = get_public_dataset(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN","3_layer_CNN"]
    pretrain_models = []
    pretrain_modelsindex = []
    for epoch in range(args.user_number):
        # 给每个local一个不同的模型结构
        tempindex = random.randint(0,1)
        pretrain_modelsindex.append(tempindex)
        tempmodel = models[modelsindex[tempindex]]()
        pretrain_models.append(tempmodel)
    train_models(device,pretrain_models,pretrain_modelsindex,train_dataset,test_dataset,args.lr,args.optimzier,args.epoch)

