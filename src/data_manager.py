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
        self.dataset = self.dm_config.dataset
        self.rank = self.config.K
        self.D = config.D
    
    def _preprocess_data(self, dataset, mean: float = 0.5, std: float = 0.5):
        data_size = dataset.data.shape[0]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
        
        xs, ys = next(iter(dataloader))
        xs = xs.view(xs.shape[0], -1)
        
        xs /= xs.std() *2 
        xs -= xs.mean()
        
        dataset = torch.utils.data.TensorDataset(xs, ys)
    
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
        


    
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        
        eigenvalues_torch = torch.from_numpy(eigenvalues).float()
        eigenvectors_torch = torch.from_numpy(eigenvectors).float()

        print("good?")
        
        num_samples = int(self.D * self.alpha)
        
        sqrtLambda = torch.diag(torch.sqrt(eigenvalues_torch))
        
        zs = torch.randn(num_samples, self.D, generator=gen)
        xs = zs @ sqrtLambda @ eigenvectors_torch.T
        
        
        # dummy labels
        ys = torch.zeros(num_samples, dtype=torch.long)
        
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
        
        
        trainloader, (_, _, eigenvalues, eigenvectors) = self.get_trainloader(bs=10_000)
        xs, _ = next(iter(trainloader))
        xs = xs.to(torch.float32)
        
        eigenvectors = torch.from_numpy(eigenvectors).to(xs.device, xs.dtype)
        
        pca_errors = []
        
        for m in range(1, max_rank + 1):
            print(f"calculting error for rank {m}")
            projections = eigenvectors[:, -m:]
            z = xs @ projections
            
            x_reconstruction = z @ projections.T

            pca_error = torch.nn.functional.mse_loss(x_reconstruction, xs) * 0.5
            
            pca_errors.append(pca_error)
        
        
        return pca_errors
            
        