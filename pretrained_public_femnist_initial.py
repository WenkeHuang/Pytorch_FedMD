import  torch
from  data_utils import  get_public_dataset
from  models import CNN_2layer_fc_model,CNN_3layer_fc_model
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from option import args_parser

if __name__ == '__main__':
    args = args_parser()
    models = {"2_layer_CNN": CNN_2layer_fc_model,  # 字典的函数类型
          "3_layer_CNN": CNN_3layer_fc_model}
    modelsindex = ["2_layer_CNN","3_layer_CNN"]
    pretrain_models = []
    pretrain_modelsindex = [1,1,1,0,1,1,0,0,1,0]
    for index,item in enumerate(pretrain_modelsindex):
        print(item)
        tempmodel = models[modelsindex[item]]()
        torch.save(tempmodel.state_dict(), 'Src/EmptyModel/LocalModel{}Type{}.pkl'.format(index, pretrain_modelsindex[index]))
