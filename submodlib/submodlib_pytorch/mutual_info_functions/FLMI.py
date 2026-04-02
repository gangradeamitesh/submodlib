from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np

class FacilityLocationMutualInformation(BaseFunction):
    def __init__(self, n , num_queries , data_sijs=None, query_sijs=None,
                 data=None, query_data=None, metric="cosine", magnificationEta=1, device=None):
        """
        Initializes the Facility Location Mutual Information Function.

        Parameters:
        - data: The ground set data points.
        - query_data: The query set data points.
        - num_neighbors: Number of neighbors to consider for mutual information calculation.
        """

        super().__init__(n=n, sijs=data_sijs, data=data, metric=metric,query_data=query_data,query_sijs=query_sijs, device=device)
        self.magnificationEta = magnificationEta
        self.effective_ground_set = None
        self.num_queries = num_queries


        validate_n(self.n)
        if self.sijs is not None:
            self.sijs = self._tensor(self.sijs)
            validate_sijs(type(self.sijs))
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float16)
            if self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            elif self.metric == "rbf":
                self.sijs = DenseSimilarity.rbf_similarity(self.data , self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        if self.query_sijs is not None:
            self.query_sijs = self._tensor(self.query_sijs)
            validate_sijs(type(self.query_sijs))
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if isinstance(self.query_data, np.ndarray):
                self.query_data = self._tensor(self.query_data, dtype=torch.float16)
            if self.metric == "euclidean":
                self.query_sijs = DenseSimilarity.euclidean_distance(self.data , self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
            elif self.metric == "rbf":
                self.query_sijs = DenseSimilarity.rbf_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided") 
        self._initialize_ground_sets()
        self._initialize_query_cap(self.query_sijs)
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()

    
    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.n, dtype=torch.float32))
        self.memoization_initialized = True


    def _initialize_query_cap(self, query_sijs):
        self.query_cap = self.magnificationEta * torch.max(query_sijs, dim=1).values

    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(self , budget=budget , stopIfZeroGain=stopIfZeroGain , stopIfNegativeGain=stopIfNegativeGain , epsilon=epsilon , verbose =verbose, show_progress=show_progress)

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
        X_tensor = torch.tensor(list(evaluate_set), dtype=torch.long)
        return torch.minimum(torch.max(self.sijs[:,X_tensor],dim=1).values,self.query_cap).sum()
    

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        if not self.memoization_initialized:
            return self.marginalGain(X, element)
        memo = self.similarity_with_nearest_in_effective_x
        candidate_sim = self.sijs[:, element]
        new_best = torch.maximum(memo, candidate_sim)
        return (torch.minimum(new_best, self.query_cap) - torch.minimum(memo, self.query_cap)).sum().item()
    

    def batchedGain(self , X):
        remaining = (~X).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            return torch.tensor(0., device=self.device), torch.tensor(-1, device=self.device)
        candidate_sims = self.sijs[:, remaining]
        new_best = torch.maximum(self.similarity_with_nearest_in_effective_x.unsqueeze(1) , candidate_sims)
        old_cap = torch.minimum(self.similarity_with_nearest_in_effective_x.unsqueeze(1),self.query_cap.unsqueeze(1))
        new_cap = torch.minimum(new_best,self.query_cap.unsqueeze(1))
        gains = (new_cap-old_cap).sum(dim=0)                
        best_gain = gains.max()
        tied_rel_indices = torch.where(gains == best_gain)[0]
        rel_idx = tied_rel_indices[-1]
        best_idx = remaining[rel_idx]        
        return best_gain, best_idx
    
    def updateBatchMemo(self , element):
        candidate = self.sijs[: , element]
        self.similarity_with_nearest_in_effective_x = torch.maximum(self.similarity_with_nearest_in_effective_x , candidate)
        #self.similarity_with_nearest_in_effective_x = candidate

    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        return torch.minimum(self.similarity_with_nearest_in_effective_x, self.query_cap).sum().item()
    

    def updateMemoization(self , X , element):
        """Update the memoization for the given set"""
        if not self.memoization_initialized or element in X:
            return
        elemenet_tensor = self._tensor(element, dtype=torch.long)
        new_similarities = self.sijs[:, elemenet_tensor]
        torch.maximum(self.similarity_with_nearest_in_effective_x, new_similarities, out=self.similarity_with_nearest_in_effective_x)        

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