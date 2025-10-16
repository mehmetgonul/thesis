from abc import ABC, abstractmethod

class Integral(ABC):
    
    @abstractmethod
    def I2(self):
        raise NotImplementedError("I2 method must be implemented by subclass")
    
    @abstractmethod
    def J2(self):
        raise NotImplementedError("J2 method must be implemented by subclass")
    
    @abstractmethod
    def I21(self):
        raise NotImplementedError("I21 method must be implemented by subclass")
    
    @abstractmethod
    def I22(self):
        raise NotImplementedError("I22 method must be implemented by subclass")
    
    @abstractmethod
    def I3(self):
        raise NotImplementedError("I3 method must be implemented by subclass")
