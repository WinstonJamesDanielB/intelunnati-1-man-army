
#code to import the dataset into your python file:


import torchvision.datasets as ds
from torchvision import transforms


fashion_mnist = ds.FashionMNIST(download=False, train=True, root="fashion_mnist_data").train_data.float()
data_transform = transforms.Compose([ transforms.Resize((28, 28)),                                    
                                         transforms.ToTensor(),
                                         transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
#for training and validation data
trainset = ds.FashionMNIST(root='fashion_mnist_data',
                                      train=True,
                                      download=True,
                                      transform=data_transform,
                          )
#for testing data

testset = ds.FashionMNIST(root='fashion_mnist_data',
                                     train=False,
                                     download=True,
                                     transform=data_transform
                                    )
