from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import torch

# load mnist data from torch database, split by batch size and take subset of data.
def load_mnist(batch_size=128, subset=10):

    data_path='/tmp/data/mnist'
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    # pull data from pytorch
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_train = utils.data_subset(mnist_train, subset)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    mnist_test = utils.data_subset(mnist_test, subset)

    # returns training and testing data as a torch Dataloader, with training data shuffled
    return DataLoader(mnist_train, batch_size=batch_size, shuffle=True), DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
