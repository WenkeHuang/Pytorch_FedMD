from torchvision import datasets,transforms

def get_public_dataset(args):
    if args.dataset =='mnist':
        data_dir ='Data/mnist/'
    apply_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root=data_dir,train=True,download=True,transform=apply_transform)
    data_test = datasets.MNIST(root=data_dir,train=False,download=True,transform=apply_transform)
    return data_train,data_test

class args:
    dataset = 'mnist'
    
get_public_dataset(args)