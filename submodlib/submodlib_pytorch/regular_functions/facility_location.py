import torch
import torch.nn.functional as F
import numpy as np

# Relative imports within the same package
from ..userValidator import validate_n, validate_mode, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
from ...logger import get_logging, enable_logging

class FacilityLocation(BaseFunction):
    """
    Facility Location function implementation based on C++ version.
    
    The function value for a set X is: sum over all master items of their maximum similarity to any item in X.
    """

    def __init__(self, n, mode="dense", sijs=None, 
                 data=None, metric="cosine",device='cpu'):
        
        super().__init__(n=n, mode=mode, sijs=sijs, data=data, metric=metric, device=device)
        self.effective_ground = None
        self.optimizer = None
    
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        self.effective_ground_set = None
        self.master_set = None
        self.n_master = None
        
        validate_n(self.n)
        validate_mode(self.mode)

        if self.sijs is not None:
            validate_sijs(type(self.sijs), self.mode)
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

    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, 
                 verbose=False, show_progress=True):
        """Maximize the function using the optimizer"""
        enable_logging()
        logger = get_logging()
        logger.info(f"Starting optimization with optimizer: {optimizer}, budget: {budget} , stopIfZeroGain: {stopIfZeroGain} , stopIfNegativeGain: {stopIfNegativeGain} , verbose: {verbose} , show_progress: {show_progress}")
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget=budget , stopIfZeroGain=stopIfZeroGain , stopIfNegativeGain=stopIfNegativeGain  , verbose=verbose , show_progress=show_progress)

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
    
    def batchedGain(self, selected_mask):
        remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(-1, device=self.device)

        memo = self.similarity_with_nearest_in_effective_x.unsqueeze(1) 
        candidates = self.sijs[:, remaining]                              

        new_best = torch.maximum(memo, candidates)
        gains = (new_best - memo).sum(dim=0)                  
        best_gain, rel_idx = gains.max(dim=0)
        best_idx = remaining[rel_idx]

        return best_gain, best_idx
    
    def updateBatchMemo(self, element):
        candidate = self.sijs[:, element]
        self.similarity_with_nearest_in_effective_x = torch.maximum(
            self.similarity_with_nearest_in_effective_x, candidate
        )

    def evaluateWithMemoization(self, evaluate_set):
        """
        Evaluate using pre-computed memoization.
        Assumes memoization is up-to-date for the given set.
        """
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        
        return torch.sum(self.similarity_with_nearest_in_effective_x).item()

    def updateMemoization(self, element):
        """
        Update memoization for the given set X.
        This is called after each greedy selection.
        """
        candidate = self.sijs[:, element]
        torch.maximum(candidate, self.similarity_with_nearest_in_effective_x, out=self.similarity_with_nearest_in_effective_x)
        

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
    

if __name__ == "__main__":
    
    data = torch.randn(1000, 1024 ,dtype=torch.float16)
    query = torch.randn(800, 1024 , dtype=torch.float16)
    obj = FacilityLocation(n=data.shape[0],data=data , metric="cosine")

    greedy_list = obj.maximize(optimizer="NaiveGreedy" , budget=BUDGET , stopIfZeroGain=False , stopIfNegativeGain =False, epsilon=False , verbose=False , show_progress=False , costs=None , costSensitiveGreedy=False)
    print("---------------")
    print(greedy_list)


    print("C++ --------------------")
    # obj = FacilityLocationFunction(n=data.shape[0], data=data, 
    #                                                 metric="cosine", 
    #                                                )
    # greedyList = obj.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
    #                           stopIfNegativeGain=False, verbose=False)
    # print(greedyList)