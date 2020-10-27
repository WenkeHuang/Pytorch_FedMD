import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # pretrained_public_mnist_initial
    parser.add_argument('--gpu',type=int,default=1,help="choose to use GPU or CPU")
    parser.add_argument('--dataset',type=str,default='mnist',help='name of dataset')
    parser.add_argument('--user_number',type=int,default=10,help='number of user join in Federated Learning')
    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
    parser.add_argument('--optimizer',type=str,default='adam',help='type of optimizer')
    parser.add_argument('--epoch',type=int,default=10,help='training epoch')

    # pretrained_public_mnist_continue
    parser.add_argument('--initialurl',type=str,default='Src\Model',help='initial url for saving initial models')
    parser.add_argument('--continue_epoch',type=int,default=10,help='epoch for continuing traning on pretrained model on mnist')

    # data_utils
    parser.add_argument('--private_dataset_index',type=str,default='Src/private_dataset_index.txt',help='in order to fix the private dataset,beacuse each client has only a few of dataset')

    # private_model_femnist_balanced
    parser.add_argument('--privateepoch',type=int,default=20,help='training epoch')
    parser.add_argument('--private_dataset',type=str,default='FEMNIST',help='private dataset for each client')
    parser.add_argument('--privateurl',type=str,default='Src\PrivateModel',help='private model location')
    parser.add_argument('--new_private_training',type=bool,default=True,help='whether train model from initial condition')
    parser.add_argument('--testBestCondition',type=bool,default=False,help='test the result with data shared')

    # Collaborative_private_model_femnist_balanced
    parser.add_argument('--new_collaborative_training',type=bool,default=False,help='whether train model from initial condition')
    parser.add_argument('--Collaborativeurl',type=str,default='Src\CollaborativeModel',help='collaborative model location')
    parser.add_argument('--output_classes',type=int,default=16,help='set the output points')
    parser.add_argument('--collaborative_epoch',type=int,default=2,help='collaborative_epoch for train on public mnist')
    # Collaborative_step
    parser.add_argument('--Communicationepoch',type=int,default=20,help='Collaobrative epoch in Step3')

    # collaborative_private_model_femnist_balanced
    parser.add_argument('--Communication_private_epoch',type=int,default=2 ,help='Local private training during colaboratiive time')

    args = parser.parse_args()

    return args