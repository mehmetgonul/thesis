import torch
from torchvision import transforms, datasets
import numpy as np

class DataManager:
    
    def __init__(self, dataset: str, rank: int, D: int, noise: float, gaussian: bool):
        
        self.dataset = dataset
        self.rank = rank
        self.D = D
        self.noise = noise
        self.gaussian = gaussian
    
    def _preprocess_data(self, dataset, mean: float = 0.5, std: float = 0.5):
        data_size = dataset.data.shape[0]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
        
        xs, ys = next(iter(dataloader))
        xs = xs.squeeze().reshape(xs.shape[0], (xs.shape[-1] ** 2))
        
        # TODO: implement a check for interpolation (usually shouldn't even be necessary)
        # xs = torch.nn.functional.interpolate(xs.unsqueeze(0), size=self.D).squeeze()
        
        xs /= xs.std()*2
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
        
        elif self.dataset == "random":
            # TODO: Ask abou this
            
            eigenvalues = torch.tensor([(2**m + 1) for m in range(1, self.rank+1)]).float()
    
            A = torch.randn(self.rank, self.D)
            c = torch.rand(self.batch_size, self.rank) * eigenvalues.unsqueeze(0).repeat(self.batch_size, 1)

            # data model
            trainset = (c @ A) + (self.noise * torch.randn(10_000, self.D) )

        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        if self.dataset != "random":
            trainset = self._preprocess_data(trainset)
        
        # this in principle only work with cifar and mnist
        if self.gaussian:
            omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(trainset)
            xs = np.random.randn(len(trainset), trainset[0][0].numel()) @ sqrt_omega
            xs = torch.from_numpy(xs).float()
            ys = trainset.tensors[1]
            trainset = torch.utils.data.TensorDataset(xs, ys)
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True, pin_memory=True)
            
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=True)
            omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(trainset)
        
        
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
        
        elif self.dataset == "random":
            
            eigenvalues = torch.tensor([(2**m + 1) for m in range(1, self.rank+1)]).float()
    
            A = torch.randn(self.rank, self.D)
            c = torch.rand(self.batch_size, self.rank) * eigenvalues.unsqueeze(0).repeat(self.batch_size, 1)

            # data model
            xs = (c @ A) + (self.noise * torch.randn(10_000, self.D) )
            ys = torch.randint(0, 10, (10_000,)) # Dummy labels
            testset = torch.utils.data.TensorDataset(xs, ys)

        else: 
            raise ValueError(f"Datamanager: self.dataset is not valid '{self.dataset}'")
        
        
        if self.dataset != "random":
            testset = self._preprocess_data(testset)
        
        if self.gaussian:
            # x ∼ N(0, omega)
            
            omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(testset)
            
            xs = np.random.randn(len(testset), testset[0][0].numel()) @ sqrt_omega
            xs = torch.from_numpy(xs).float()
            ys = testset.tensors[1]
            testset = torch.utils.data.TensorDataset(xs, ys)

            testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                shuffle=False)
            
        else:
            omega, sqrt_omega, eigenvalues, eigenvectors = self._compute_covariance(testset)
            testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                                shuffle=False)
            
        
        return testloader, (omega, sqrt_omega, eigenvalues, eigenvectors)

    def _compute_covariance(self, dataset):
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        xs , _  = next(iter(loader))
        omega = (xs.t()@xs/xs.shape[0]).cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        # eigenvalues += np.abs(np.min(eigenvalues)) # for numerical stability
        sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
        return omega, sqrt_omega, eigenvalues, eigenvectors