import torch
from src.config import RunConfig
import numpy as np
from torch.utils.data import DataLoader, Dataset
from skrmt.ensemble import WishartEnsemble


SPIKED_WISHART = "spiked_wishart"


class GaussianCovarianceDataset(Dataset):
    """Draw fresh Gaussian inputs lazily from a fixed covariance."""

    def __init__(self, num_samples, sqrt_omega, seed):
        self.num_samples = int(num_samples)
        self.sqrt_omega = torch.as_tensor(sqrt_omega, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _index):
        z = torch.randn(
            self.sqrt_omega.shape[0],
            generator=self.generator,
            dtype=self.sqrt_omega.dtype,
        )
        return z @ self.sqrt_omega, 0


class DataManager:
    def __init__(self, seed: int, config: RunConfig):
        self.seed = seed
        self.config = config
        self.alpha = config.alpha
        self.dataset = config.dataset
        self.rank = config.K
        self.D = config.D

        if self.dataset != SPIKED_WISHART:
            raise ValueError(f"Unsupported dataset {self.dataset!r}. Expected {SPIKED_WISHART!r}.")
        if config.wishart_samples is None:
            raise ValueError("spiked_wishart requires 'wishart_samples'")
        if not config.spike_multipliers:
            raise ValueError("spiked_wishart requires 'spike_multipliers'")

        M = int(config.wishart_samples)
        if M <= 0:
            raise ValueError("wishart_samples must be positive")

        spike_multipliers = np.asarray(config.spike_multipliers, dtype=float)
        if np.any(spike_multipliers <= 0):
            raise ValueError("spike_multipliers must be positive")

        wishart = WishartEnsemble(
            p=self.D,
            n=M,
            beta=1,
            random_state=self.seed,
        )
        A = wishart.matrix / M

        rng = np.random.default_rng(self.seed + 1)
        U, _ = np.linalg.qr(
            rng.standard_normal((self.D, len(spike_multipliers))),
            mode="reduced",
        )
        spike_strengths = spike_multipliers * np.sqrt(self.D)
        omega = A + (U * spike_strengths) @ U.T

        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sqrt_omega = np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T
        self.spiked_statistics = (
            omega,
            sqrt_omega,
            eigenvalues,
            eigenvectors,
        )

    def get_trainloader(self):
        gaussian_set = GaussianCovarianceDataset(
            num_samples=self.D * self.alpha,
            sqrt_omega=self.spiked_statistics[1],
            seed=self.seed + 2,
        )
        return DataLoader(gaussian_set, batch_size=1, shuffle=False), self.spiked_statistics

    def get_ode_statistics(self):
        """Return the reference covariance spectrum without making samples."""
        return self.spiked_statistics

    def get_testloader(self):
        testset = GaussianCovarianceDataset(
            num_samples=10_000,
            sqrt_omega=self.spiked_statistics[1],
            seed=self.seed + 3,
        )
        return DataLoader(testset, batch_size=len(testset), shuffle=False), self.spiked_statistics

    def pca_errors(self, max_rank=None):
        if max_rank is None:
            max_rank = self.rank

        eigenvalues = self.spiked_statistics[2]
        max_rank = min(max_rank, len(eigenvalues))
        return [
            0.5 * float(eigenvalues[:-m].sum()) / len(eigenvalues)
            for m in range(1, max_rank + 1)
        ]
