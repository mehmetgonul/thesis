import torch
from src.config import RunConfig
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader

class DataManager:
    
    def __init__(self, seed: int, config: RunConfig):
        self.seed = seed
        self.config = config
        self.alpha = config.alpha
        self.dataset = self.config.dataset
        self.rank = self.config.K
        self.D = config.D
    
    def _preprocess_data(self, dataset, mean=None, std=None, return_stats=False):
        data_size = dataset.data.shape[0]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
        
        xs, ys = next(iter(dataloader))
        xs = xs.view(xs.shape[0], -1)
        
        if self.config.preprocess_data:
            if std is None:
                std = xs.std()
            xs /= std * 2
            if mean is None:
                mean = xs.mean()
            xs -= mean
        else:
            if mean is None:
                mean = torch.zeros((), dtype=xs.dtype, device=xs.device)
            if std is None:
                std = torch.ones((), dtype=xs.dtype, device=xs.device)
        
        dataset = torch.utils.data.TensorDataset(xs, ys)
    
        if return_stats:
            return dataset, mean, std
        return dataset
    
    def get_trainloader(self):
        
        if self.dataset == "fashion_mnist":
            input_transforms = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.FashionMNIST(root="./data/fashionMNIST", train=True,
                                download=True, 
                                transform=input_transforms)
        
        elif self.dataset == "cifar-10":
            input_transforms = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
            trainset = datasets.CIFAR10(root="./data/cifar-10/", train=True,
                            download=True, 
                            transform=input_transforms)

        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        trainset = self._preprocess_data(trainset)
        omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(trainset)

        loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        xs, _ = next(iter(loader))
        

        num_samples = int(self.D * self.alpha)

        
        
        # dummy labels
        ys = torch.zeros(num_samples, dtype=torch.long)

        if num_samples > len(xs):
            repeats = (num_samples // len(xs)) + 1
            xs = xs.repeat(repeats, 1)[:num_samples]
        else:
            xs = xs[:num_samples]
        trainset = torch.utils.data.TensorDataset(xs, ys)
        
        trainloader = DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
        )
        print("good till here")    
            
        return trainloader, (omega, sqrt_omega, eigenvalues, eigenvectors)


    # TODO: compute from op definition
    def get_testloader(self):
        
        if self.dataset == "fashion_mnist":
            input_transforms = transforms.Compose([transforms.ToTensor()])
            testset = datasets.FashionMNIST(root="./data/fashionMNIST", train=False,
                                download=True, 
                                transform=input_transforms)
        
        elif self.dataset == "cifar-10":
            input_transforms = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor()])
            testset = datasets.CIFAR10(root="./data/cifar-10/", train=False,
                            download=True, 
                            transform=input_transforms)
        
        
        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        
        testset = self._preprocess_data(testset)
        omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(testset)
            
    
        
        testloader = DataLoader(
            testset,
            batch_size=len(testset),
        )
        
        xs , _  = next(iter(testloader))
        print(f"the norm^2 of my data is {xs.norm(dim=1).mean()**2 }")
        print(f"std of data is {xs.std()}")
        
        return testloader, (omega, sqrt_omega, eigenvalues, eigenvectors)


    def get_probe_loaders(self, batch_size: int = 256):
        if self.dataset == "fashion_mnist":
            input_transforms = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.FashionMNIST(root="./data/fashionMNIST", train=True,
                                download=True,
                                transform=input_transforms)
            testset = datasets.FashionMNIST(root="./data/fashionMNIST", train=False,
                                download=True,
                                transform=input_transforms)

        elif self.dataset == "cifar-10":
            input_transforms = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
            trainset = datasets.CIFAR10(root="./data/cifar-10/", train=True,
                            download=True,
                            transform=input_transforms)
            testset = datasets.CIFAR10(root="./data/cifar-10/", train=False,
                            download=True,
                            transform=input_transforms)

        else:
            raise ValueError(f"Probe loaders unsupported for dataset '{self.dataset}'")

        trainset, mean, std = self._preprocess_data(trainset, return_stats=True)
        testset = self._preprocess_data(testset, mean=mean, std=std)

        return (
            DataLoader(trainset, batch_size=batch_size, shuffle=True),
            DataLoader(testset, batch_size=batch_size, shuffle=False),
        )


    def _compute_covariance(self, dataset):
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        xs , _  = next(iter(loader))
        omega = (xs.t()@xs/xs.shape[0]).cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
        return omega, sqrt_omega, eigenvalues, eigenvectors
    
    
    
    def pca_errors(self, max_rank=None):
        
        if max_rank is None:
            max_rank = self.rank
        
        if self.dataset == "fashion_mnist":
            input_transforms = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.FashionMNIST(root="./data/fashionMNIST", train=True,
                                download=True,
                                transform=input_transforms)
        elif self.dataset == "cifar-10":
            input_transforms = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
            trainset = datasets.CIFAR10(root="./data/cifar-10/", train=False,
                            download=True,
                            transform=input_transforms)
        else:
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")

        trainset = self._preprocess_data(trainset)
        _, _, eigenvalues, _ = self._compute_covariance(trainset)
        max_rank = min(max_rank, len(eigenvalues))

        return [0.5 * float(eigenvalues[:-m].sum()) / len(eigenvalues) for m in range(1, max_rank + 1)]
            
        
