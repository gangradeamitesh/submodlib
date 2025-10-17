from abc import ABC, abstractmethod

class BaseOptimizer:

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def optimize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        pass