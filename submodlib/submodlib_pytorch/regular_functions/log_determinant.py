from pickle import NONE
from submodlib.submodlib_pytorch.base_function import BaseFunction
from submodlib.submodlib_pytorch.optimizers.optimizer_factory import OptimizerFactory
import torch
from submodlib.submodlib_pytorch.userValidator import validate_n, validate_mode, validate_sep_rep , validate_sijs
from submodlib.submodlib_pytorch.cal_simi_kernel import DenseSimilarity
import numpy as np

class LogDeterminant(BaseFunction):
    
    def __init__(self, n, mode="dense", sijs=None, 
                 data=None, num_clusters=None, cluster_labels=None, 
                 metric="cosine") -> None:
        super().__init__(n, mode, sijs, data, num_clusters, metric, cluster_labels)
        self.optimizer = None
        
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        
        self.effective_ground_set = None
        self._initialize_ground_sets()
        validate_n(self.n)
        validate_mode(self.mode)
        #validate_sep_rep(self.separate_rep, self.mode, self.n_rep)
        if self.sijs is not None:
            validate_sijs(type(self.sijs))
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)
            
            if self.mode == "dense" and self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
    
    def _initialize_ground_sets(self):
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)


    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        
        curr_val = self.evaluate(X)
        X_new = X | {element}
        new_val = self.evaluate(X_new)
        return new_val - curr_val


    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        if not evaluate_set:
            return 0.0
        if not evaluate_set:
            return 0.0
        X_list = list(evaluate_set)
        X_tensor = torch.tensor(X_list, dtype=torch.long)
        # effective_ground_tensor = torch.tensor(list(self.effective_ground_set), dtype=torch.long)
        det_X_tensor = self.sijs[X_tensor][:, X_tensor]
        return torch.logdet(det_X_tensor)

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        gain = 0.0
        element_tensor = self._tensor([element], dtype=torch.long)
        new_similarities = self.sijs[:, element_tensor]
        gain_tensor = torch.maximum(new_similarities - self.similarity_with_nearest_in_effective_x, torch.tensor(0.0))
        gain = torch.sum(gain_tensor).item()
        return gain

    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        
        return torch.sum(self.similarity_with_nearest_in_effective_x).item()


    def updateMemoization(self , X,element):
        """Update the memoization for the given set"""
        if not self.memoization_initialized or element in X:
            return
        element_tensor = self._tensor(element, dtype=torch.long)
        new_similarity = self.sijs[:, element_tensor]
        torch.maximum(new_similarity, self.similarity_with_nearest_in_effective_x, out=self.similarity_with_nearest_in_effective_x)


    def clearMemoization(self):
        """Clear the memoization"""
        if self.memoization_initialized:
            self.similarity_with_nearest_in_effective_x.zero_()

    def setMemoization(self , X):
        """Set the memoization for the given set"""
        if not self.memoization_initialized:
            return
        
        self.clearMemoization()
        running = set()
        for ele in X:
            self.updateMemoization(running , ele)
            running.add(ele)    

    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        return self.effective_ground_set