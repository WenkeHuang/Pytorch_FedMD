from option import args_parser
from  collaborative_private_model_femnist_balanced import collaborative_private_model_femnist_train
from  collaborative_train_balanced_mnist import collaborative_private_model_mnist_train
from collaborative_model_femnist_Accuracy import test_accuracy_collaborativemodel
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator #从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

def transpose( matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix

if __name__ == '__main__':
    args = args_parser()
    accuracy = []
    for item in range(args.Communicationepoch):
        print('This is {} time communcation'.format(item))
        eachround_accuracy = []
        eachround_accuracy = test_accuracy_collaborativemodel(args)
        collaborative_private_model_mnist_train(args)
        collaborative_private_model_femnist_train(args)
        accuracy.append(eachround_accuracy)
    eachround_accuracy = []
    eachround_accuracy = test_accuracy_collaborativemodel(args)
    accuracy.append(eachround_accuracy)
    accuracy = transpose(accuracy)

    for i, val in enumerate(accuracy):
        print(val)
        plt.plot(range(len(val)), val,label='model :'+str(i))
    plt.legend(loc='best')
    plt.title('communication_round_with_accuracy_on_femnist')
    plt.xlabel('Communication roudn')
    plt.ylabel('Accuracy on FEMNIST')
    x_major_locator = MultipleLocator(1)# 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()# ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)# 把x轴的主刻度设置为1的倍数
    plt.xlim(0, args.Communicationepoch)
    plt.savefig('Src/Figure/communication_round_with_accuracy_on_femnist.png')
    plt.show()
    print('End Private Training')