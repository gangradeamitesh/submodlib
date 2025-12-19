from abc import ABC , abstractmethod
import torch
from submodlib.runtime import get_default_device

class BaseFunction(ABC):

    def __init__(self,n , mode="dense",sijs=None,data=None,num_clusters=None,cluster_label=None,metric="cosine",cluster_labels=None,query_data=None,query_sijs=None , device = None) -> None:
        self.device = torch.device(device) if device else get_default_device()
        self.n = n
        self.mode = mode
        self.metric = metric
        self.data = self._tensor(data)
        self.clusters = None
        self.cluster_sijs = None
        self.cluster_map = None
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters
        self.sijs = sijs
        if query_data is not None:
            self.query_data = self._tensor(query_data)
        self.query_sijs = query_sijs
        
    
    def _tensor(self, val , **kwargs):
        return torch.as_tensor(val , device=self.device, **kwargs)
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.data.shape[0], dtype=torch.float32))
        self.memoization_initialized = True

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