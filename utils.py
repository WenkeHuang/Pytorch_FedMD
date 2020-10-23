import torch
import os

def get_model_list(url,modelsindex,models):
    model_list = []
    model_type_list = []
    filePath = url
    for root, dirs, files in os.walk(filePath, topdown=False):
        for name in files:
            model_type_list.append(int(name[name.find('Type')+4]))
            net = models[modelsindex[int(name[name.find('Type')+4])]]()
            net.load_state_dict(torch.load(os.path.join(root, name)))
            model_list.append(net)
    return model_list,model_type_list

def get_model_list_test_acuracy(url,modelsindex,models):
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