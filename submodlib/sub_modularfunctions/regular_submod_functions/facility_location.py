import torch
import torch.nn.functional as F
import numpy as np

# Relative imports within the same package
from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory

class FacilityLocation(BaseFunction):
    """
    Facility Location function implementation based on C++ version.
    
    The function value for a set X is: sum over all master items of their maximum similarity to any item in X.
    """

    def __init__(self, n, mode="dense", sijs=None, 
                 data=None, data_rep=None, num_clusters=None, cluster_labels=None, metric="cosine", ground_set=None,device='cpu'):
        
        super().__init__(n=n, mode=mode, sijs=sijs, data=data,cluster_label=cluster_labels , num_clusters=num_clusters, metric=metric,device=device)
        self.data_rep = data_rep
        self.effective_ground = None
        self.optimizer = None
        
        self.ground_set = ground_set        
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        self.effective_ground_set = None
        self.master_set = None
        self.n_master = None
        
        validate_n(self.n)
        validate_mode(self.mode)
        #validate_sep_rep(self.mode, self.n_rep)

        if self.sijs is not None:
            validate_sijs(type(self.sijs), self.mode)
            #if self.separate_rep == True:
            #    if self.data.shape[1] != self.data_rep.shape[1]:
            #        raise Exception("ERROR: Data and Representation have different dimensions")
            #if self.data is not None or self.data_rep is not None:
            #    print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)
            
            if self.mode == "dense" and self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data, self.data)
                
            elif self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data, self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        
        self._initialize_ground_sets()
        self._initialize_memoization()

    def _initialize_ground_sets(self):
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)
        #self.n_master = len(self.effective_ground_set)

    def _initialize_memoization(self):
        """Initialize memoization structures"""
        
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.n, dtype=torch.float32))
        self.memoization_initialized = True

    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)

    def evaluate(self, evaluate_set):
        if not evaluate_set:
            return 0.0
        X_list = list(evaluate_set)
        X_tensor = self._tensor(torch.tensor(X_list, dtype=torch.long))
        max_similarities = torch.max(self.sijs[:, X_tensor], dim=1)[0]  # Shape: [n_master]
        return torch.sum(max_similarities).item()
    
    def marginalGain(self, X, element):
        """
        Compute marginal gain of adding element to set X.
        This is the difference in function value when element is added.
        """
        if element in X:
            return 0.0
        

        if element not in self.effective_ground_set:
            return 0.0
        
        current_val = self.evaluate(X)
        X_new = X | {element}  # Use set union
        new_val = self.evaluate(X_new)
        return new_val - current_val

    def marginalGainWithMemoization(self, X, element):
        """
        Compute marginal gain using memoization for efficiency.
        This is the key optimization that makes greedy algorithms fast.
        """
        if element in X:
            return 0.0
        

        if element not in self.effective_ground_set:
            return 0.0
        
        if not self.memoization_initialized:
            return self.marginalGain(X, element)
        
        gain = 0.0
        element_tensor = self._tensor(element, dtype=torch.long)
        new_similarities = self.sijs[:, element_tensor]
        gain_tensor = torch.maximum(new_similarities - self.similarity_with_nearest_in_effective_x, torch.tensor(0.0))
        gain = torch.sum(gain_tensor).item()
        return gain

    def evaluateWithMemoization(self, evaluate_set):
        """
        Evaluate using pre-computed memoization.
        Assumes memoization is up-to-date for the given set.
        """
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        
        return torch.sum(self.similarity_with_nearest_in_effective_x).item()

    def updateMemoization(self, X, element):
        """
        Update memoization for the given set X.
        This is called after each greedy selection.
        """
        if not self.memoization_initialized or element in X:
            return
        element_tensor = self._tensor(element, dtype=torch.long)
        new_similarity = self.sijs[:, element_tensor]
        torch.maximum(new_similarity, self.similarity_with_nearest_in_effective_x, out=self.similarity_with_nearest_in_effective_x)
        

    def clearMemoization(self):
        """Clear all memoization data"""
        if self.memoization_initialized:
            self.similarity_with_nearest_in_effective_x.zero_()

    def setMemoization(self, X):
        """
        Set memoization for a given set X.
        This initializes memoization as if X was the current set.
        """
        if not self.memoization_initialized:
            return
        
        self.clearMemoization()
        running = set()
        for ele in X:
            self.updateMemoization(running , ele)
            running.add(ele)
        

    def getEffectiveGroundSet(self):
        """
        Return the effective ground set (actual set, not size).
        This matches the C++ implementation.
        """
        return self.effective_ground_set  # Return a copy to prevent external modification