from abc import ABC , abstractmethod

class BaseFunction(ABC):

    def __init__(self,n , mode="dense",sijs=None,data=None,num_clusters=None,cluster_label=None,metric="cosine",cluster_labels=None,query_data=None,query_sijs=None) -> None:
        self.n = n
        self.mode = mode
        self.metric = metric
        self.data = data
        self.clusters = None
        self.cluster_sijs = None
        self.cluster_map = None
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters
        self.sijs = sijs
        self.query_data = query_data
        self.query_sijs = query_sijs
        

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
    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        pass
    @abstractmethod
    def evaluateWithMemoization(self , evaluate_set):
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