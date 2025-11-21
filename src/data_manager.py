import torch
from src.config import RunConfig
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

class DataManager:
    
    def __init__(self, seed: int, config: RunConfig):
        self.seed = seed
        self.config = config
        self.dm_config = config.dm_config
        self.alpha = config.alpha
        self.dataset = self.dm_config.dataset
        self.rank = self.dm_config.rank
        self.D = config.D
        self.noise = self.dm_config.noise
        self.gaussian = self.dm_config.gaussian
        self.online = (config.training_method == "online") # aligned with the config
    
    def _preprocess_data(self, dataset, mean: float = 0.5, std: float = 0.5):
        data_size = dataset.data.shape[0]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
        
        xs, ys = next(iter(dataloader))
        xs = xs.view(xs.shape[0], -1)
        
        xs -= xs.mean()
        xs /= xs.std()*2
        
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
        
        elif self.dataset == "synthetic":
            
            # eigenvalues = torch.tensor([(2**m + 1) for m in range(1, self.rank+1)]).float()
            v = []
            lam = []
            for i in range(self.rank):
                lam.append(self.D * (i + 1))
                temp = np.zeros(self.D)
                temp[i] = 1
                v.append(temp)
            
            v = np.array(v)
            
            cov = np.identity(self.D)
            
            for i in range(self.rank):
                cov += lam[i] * (v.T[i] @ (v)[i])

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
            print("------------------------------------------------")
            print(cov)
            print("------------------------------------------------")
            return cov, sqrt_omega, eigenvalues, eigenvectors

        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        if self.dataset != "synthetic":
            trainset = self._preprocess_data(trainset)
        
        # this in principle only work with cifar and mnist
        omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(trainset)
        if self.gaussian:
            xs = np.random.randn(len(trainset), trainset[0][0].numel()) @ sqrt_omega
            xs = torch.from_numpy(xs).float()
            ys = trainset.tensors[1]
            trainset = torch.utils.data.TensorDataset(xs, ys)
            
        if self.online:
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            random_sampler = RandomSampler(
                trainset,
                replacement=True,
                num_samples= self.D * self.alpha,
                generator=gen
            )
            
            trainloader = DataLoader(
                trainset,
                batch_size=1,
                sampler=random_sampler,
                # num_workers=2,
                # prefetch_factor=1,
                # pin_memory=True,
            )
        
        else:
            trainloader = DataLoader(
                trainset,
                batch_size=1,
            )
        
        
        return trainloader, (omega, sqrt_omega, eigenvalues, eigenvectors)

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
        
        elif self.dataset == "synthetic":
            
            # eigenvalues = torch.tensor([(2**m + 1) for m in range(1, self.rank+1)]).float()
            v = []
            lam = []
            for i in range(self.rank):
                lam = self.D * (i + 1)
                temp = np.zeros(self.D)
                temp[i] = 1
                v.append(temp)
            
            cov = np.identity(self.D)
            
            for i in range(self.rank):
                cov += lam[i] * v[i] * (v.T)[i]

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
            return cov, sqrt_omega, eigenvalues, eigenvectors

        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        
        if self.dataset != "synthetic":
            testset = self._preprocess_data(testset)
        
        omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(testset)
        if self.gaussian:
            xs = np.random.randn(len(testset), testset[0][0].numel()) @ sqrt_omega
            xs = torch.from_numpy(xs).float()
            ys = testset.tensors[1]
            testset = torch.utils.data.TensorDataset(xs, ys)
            
        
        testloader = DataLoader(
            testset,
            batch_size=len(testset),
            # num_workers=2,
            # prefetch_factor=1,
            # pin_memory=True,
        )
        
        
        return testloader, (omega, sqrt_omega, eigenvalues, eigenvectors)


    def _compute_covariance(self, dataset):
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        xs , _  = next(iter(loader))
        omega = (xs.t()@xs/xs.shape[0]).cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
        return omega, sqrt_omega, eigenvalues, eigenvectors