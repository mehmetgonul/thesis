"""Gaussian moments used by the loss and order-parameter dynamics."""

from abc import ABC, abstractmethod


class Integral(ABC):
    @abstractmethod
    def I2(self, C):
        """E[g(X) g(Y)] for centered Gaussian variables with covariance C."""

    @abstractmethod
    def J2(self, C):
        """E[X g(Y)]."""

    @abstractmethod
    def I21(self, C):
        """E[g'(X) X Y]."""

    @abstractmethod
    def I22(self, C):
        """E[g'(X) Y²]."""

    @abstractmethod
    def I3(self, C):
        """E[g'(X) Y g(Z)]."""
