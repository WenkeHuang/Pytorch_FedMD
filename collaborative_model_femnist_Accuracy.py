from  data_utils import  get_public_dataset,get_private_dataset_balanced
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from utils import get_model_list_test_acuracy


def test_accuracy_collaborativemodel(args):
    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, test_dataset = get_private_dataset_balanced(args)
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
              "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN", "3_layer_CNN"]
    model_list = get_model_list_test_acuracy(args.Collaborativeurl, modelsindex, models)
    accuracy_list = []
    for n, model in enumerate(model_list):
        print('Test accuracy of Local Model {}'.format(n))
        model.to(device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.NLLLoss().to(device)

        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct / total
        accuracy_list.append(accuracy)
    print(accuracy_list)
    return accuracy_list

from option import args_parser
if __name__ == '__main__':
    args = args_parser()
    test_accuracy_collaborativemodel(args)
