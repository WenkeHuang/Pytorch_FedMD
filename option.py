import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # pretrained_public_mnist_initial
    parser.add_argument('--gpu',type=int,default=1,help="choose to use GPU or CPU")
    parser.add_argument('--dataset',type=str,default='mnist',help='name of dataset')
    parser.add_argument('--user_number',type=int,default=10,help='number of user join in Federated Learning')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--optimzier',type=str,default='sgd',help='type of optimizer')
    parser.add_argument('--epoch',type=int,default=10,help='training epoch')

    # pretrained_public_mnist_continue
    parser.add_argument('--initialurl',type=str,default='Src\Model',help='initial url for saving initial models')
    parser.add_argument('--continue_epoch',type=int,default=10,help='epoch for continuing traning on pretrained model on mnist')


    args = parser.parse_args()
    return args