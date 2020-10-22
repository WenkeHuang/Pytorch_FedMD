from  data_utils import  get_public_dataset,get_private_dataset_balanced
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import os
import torch
from torch.utils.data import DataLoader
from torch import nn

class args:
    gpu =1
    dataset ='mnist'
    url = 'Src\CollaborativeModel'
    private_dataset = 'FEMNIST'

def get_model_list(url,modelsindex,models):
    model_list = []
    filePath = url
    for root, dirs, files in os.walk(filePath, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            print(name[name.find('Type')+4])
            net = models[modelsindex[int(name[name.find('Type')+4])]]()
            net.load_state_dict(torch.load(os.path.join(root, name)))
            model_list.append(net)
    return model_list
if __name__ == '__main__':
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    device ='cuda' if args.gpu else 'cpu'

    train_dataset,test_dataset = get_private_dataset_balanced(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN","3_layer_CNN"]
    model_list = get_model_list(args.url,modelsindex,models)
    accuracy_list = []
    for n, model in enumerate(model_list):
        print('Test accuracy of Local Model {}'.format(n))
        model.to(device)
        model.eval()
        loss,total,correct = 0.0,0.0,0.0
        criterion = nn.NLLLoss().to(device)

        testloader = DataLoader(test_dataset,batch_size=64,shuffle=False)
        for batch_idx,(images,labels) in enumerate(testloader):
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs,labels)
            _,pred_labels = torch.max(outputs,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels,labels)).item()
            total += len(labels)
        accuracy = correct/total
        accuracy_list.append(accuracy)
    print(accuracy_list)