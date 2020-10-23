from option import args_parser
from  collaborative_private_model_femnist_balanced import collaborative_private_model_femnist_train
from  collaborative_train_balanced_mnist import collaborative_private_model_mnist_train
from collaborative_model_femnist_Accuracy import test_accuracy_collaborativemodel
import matplotlib.pyplot as plt
if __name__ == '__main__':
    args = args_parser()
    accuracy = []
    for item in range(args.Communicationepoch):
        eachround_accuracy = []
        eachround_accuracy = test_accuracy_collaborativemodel(args)
        collaborative_private_model_mnist_train(args)
        collaborative_private_model_femnist_train(args)
        accuracy.append()

    for i, val in enumerate(accuracy):
        print(val)
        plt.plot(range(len(val)), val)
    plt.xlabel('Communication roudn')
    plt.ylabel('Accuracy on FEMNIST')
    plt.savefig('Src/Figure/communication_round_with_accuracy_on_femnist.png')
    plt.show()
    print('End Private Training')