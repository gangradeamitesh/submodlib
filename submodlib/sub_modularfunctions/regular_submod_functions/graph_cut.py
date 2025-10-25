from pickle import NONE
from submodlib.sub_modularfunctions.base_function import BaseFunction
from submodlib.sub_modularfunctions.optimizers.optimizer_factory import OptimizerFactory
import torch
from submodlib.sub_modularfunctions.userValidator import validate_n, validate_mode, validate_sep_rep , validate_sijs
from submodlib.sub_modularfunctions.cal_simi_kernel import DenseSimilarity
from submodlib import GraphCutFunction

"""TODO: To implement the lambda functionlity for graph cut"""
class GraphCut(BaseFunction):

    def __init__(self, n, mode="dense",lambda_val=0.1,mgsijs=None,ggsijs=None,data=None,
                 metric="cosine") -> None:    
        super().__init__(n=n, mode=mode,metric=metric , sijs=ggsijs, data=data)

        self.lambda_val = lambda_val
        
        self.effective_ground = None
        
        self.optimizer = None
                
        # Memoization variables (similar to C++ version)
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        
        
        self.effective_ground_set = None
        self._initialize_ground_sets()
        validate_n(self.n)
        validate_mode(self.mode)
        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
                raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")
        
        if self.sijs is not None:
            validate_sijs(type(self.sijs), self.mode, self.num_neighbors, self.separate_rep)
            if self.separate_rep == True:
                if self.data.shape[1] != self.data_rep.shape[1]:
                    raise Exception("ERROR: Data and Representation have different dimensions")
            if self.data is not None or self.data_rep is not None:
                print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = torch.tensor(self.data, dtype=torch.float32)
            
            if self.mode == "dense" and self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif  self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        self._initialize_memoization()
    def _initialize_memoization(self):
        """Initialize memoization structures"""
        
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.n, dtype=torch.float32))
        self.memoization_initialized = True
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)
    
    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)
    
    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        current_val = self.evaluate(X)
        X_new = X | {element}
        new_val = self.evaluate(X_new)
        return new_val - current_val
    
    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        if not evaluate_set:
            return 0.0
        
        X_list = list(evaluate_set)
        X_tensor = torch.tensor(X_list, dtype=torch.long)
        effective_ground_tensor = self.getEffectiveGroundSet()
        representation_term = self.sijs[:,X_tensor].sum()
        diversity_term = self.sijs[X_tensor][: ,X_tensor].sum()
        return representation_term - self.lambda_val * diversity_term


    
    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        
        # Compute the gain using memoization
        new_similarities = self.sijs[:, element]
        gain_tensor = torch.maximum(new_similarities - self.similarity_with_nearest_in_effective_x, torch.tensor(0.0))
        gain = torch.sum(gain_tensor).item()
        return gain
    
    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        
        return torch.sum(self.similarity_with_nearest_in_effective_x).item()
    
    def updateMemoization(self , X, element):
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
