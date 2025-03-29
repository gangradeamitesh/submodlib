from abc import ABC , abstractmethod

class BaseFunction(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        pass
    @abstractmethod
    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        pass
    @abstractmethod
    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        pass 
    @abstractmethod
    def maginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        pass
    @abstractmethod
    def evalauteWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        pass
    @abstractmethod
    def updateMemoization(self , X):
        """Update the memoization for the given set"""
        pass
    @abstractmethod
    def clearMemoization(self):
        """Clear the memoization"""
        pass
    @abstractmethod
    def setMemoization(self , X):
        """Set the memoization for the given set"""
        pass
    @abstractmethod
    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        pass