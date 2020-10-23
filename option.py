import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # pretrained_public_mnist_initial
    parser.add_argument('--gpu',type=int,default=1,help="choose to use GPU or CPU")
    parser.add_argument('--dataset',type=str,default='mnist',help='name of dataset')
    parser.add_argument('--user_number',type=int,default=10,help='number of user join in Federated Learning')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--optimizer',type=str,default='sgd',help='type of optimizer')
    parser.add_argument('--epoch',type=int,default=10,help='training epoch')

    # pretrained_public_mnist_continue
    parser.add_argument('--initialurl',type=str,default='Src\Model',help='initial url for saving initial models')
    parser.add_argument('--continue_epoch',type=int,default=10,help='epoch for continuing traning on pretrained model on mnist')

    # private_model_femnist_balanced
    parser.add_argument('--privateepoch',type=int,default=20,help='training epoch')
    parser.add_argument('--private_dataset',type=str,default='FEMNIST',help='private dataset for each client')
    parser.add_argument('--privateurl',type=str,default='Src\PrivateModel',help='private model location')
    parser.add_argument('--new_private_training',type=bool,default=True,help='whether train model from initial condition')

    # Collaborative_private_model_femnist_balanced
    parser.add_argument('--new_collaborative_training',type=bool,default=False,help='whether train model from initial condition')
    parser.add_argument('--Collaborativeurl',type=str,default='Src\CollaborativeModel',help='collaborative model location')
    parser.add_argument('--collaborative_epoch',type=int,default=3,help='collaborative_epoch for train on public mnist')

    # Collaborative_step
    parser.add_argument('--Communicationepoch',type=int,default=3,help='Collaorative epoch in Step3')


    args = parser.parse_args()


    return args