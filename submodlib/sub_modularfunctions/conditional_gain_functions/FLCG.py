from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np

class FacilityLocationConditionalGain(BaseFunction):
    
    def __init__(self, n=None,num_privates=None, data_sijs=None ,private_sijs=None, data=None, privateData=None, metric="cosine",privacyHardness=1):
        super().__init__(n=n,num_privates=num_privates,data=data,privateData=privateData, metric=metric)
        self.privacyHardness = privacyHardness
        self.data_sijs = data_sijs
        self.private_sijs = private_sijs

        if self.data_sijs is None and metric == "cosine":
            self.data_sijs = DenseSimilarity().cosine_similarity(self.data, self.data)
        if self.private_sijs is None and self.num_privates > 0:
            self.private_sijs = DenseSimilarity().cosine_similarity(self.data, self.privateData)
        if self.data_sijs is None and metric == "euclidean":
            self.data_sijs = DenseSimilarity().euclidean_distance(self.data, self.data)
        if self.private_sijs is None and self.num_privates > 0:
            self.private_sijs = DenseSimilarity().euclidean_distance(self.data, self.privateData)
        
            
        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")
        if self.num_privates < 0:
            raise Exception("ERROR: Number of private data points must be >= 0")
        
        self._initialize_ground_sets()
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)
    
    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        pass
    
    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        if not evaluate_set:
            return 0.0
        pass
    
    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        pass
    
    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        pass
    
    def updateMemoization(self , X):
        """Update the memoization for the given set"""
        pass
    def clearMemoization(self):
        """Clear the memoization"""
        pass

    def setMemoization(self , X):
        """Set the memoization for the given set"""
        pass

    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        pass