from submodlib.sub_modularfunctions.base_function import BaseFunction
import userValidator 
from userValidator import validate_n, validate_sep_rep , validate_sijs
import torch
from cal_simi_kernel import DenseSimilarity
import numpy as np
from optimizers import OptimizerFactory

class FacilityLocationVariantMutualInformation(BaseFunction):
    def __init__(self, n , num_queries , data_sijs=None, query_sijs=None,
                 data=None, query_data=None, metric="cosine", queryDiversityEta=1):
        """
        Initializes the Facility Location Mutual Information Function.

        Parameters:
        - data: The ground set data points.
        - query_data: The query set data points.
        - num_neighbors: Number of neighbors to consider for mutual information calculation.
        """

        super().__init__(n=n, sijs=data_sijs, data=data, metric=metric,query_data=query_data,query_sijs=query_sijs)
        self.queryDiversityEta = queryDiversityEta
        self.effective_ground_set = None
        self.query_cap = None
        self.num_queries = num_queries



        validate_n(self.n)
        if self.sijs is not None:
            """TODO: Validate data_sijs"""
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)
            
            if self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        if self.query_sijs is not None:
            """TODO : Validate query_sijs"""
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if isinstance(self.query_data, np.ndarray):
                self.query_data = self._tensor(self.query_data, dtype=torch.float32)
            if self.metric == "euclidean":
                self.query_sijs = DenseSimilarity.euclidean_distance(self.data , self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided") 
        self._initialize_ground_sets()
        self._initialize_query_cap(self.query_sijs)
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()

    def _initialize_query_cap(self, query_sijs):
        self.query_cap = self.queryDiversityEta * torch.sum(torch.max(query_sijs, dim=1)).values
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
            # Create ground set with items 0 to n-1 (like C++ lines 28-32)
        self.effective_ground_set = set(range(self.n))

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
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
        X_tensor = self._tensor(evaluate_set, dtype=torch.long)
        return torch.sum(torch.max(self.sijs[:,X_tensor])).values + self.query_cap
    


    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        # gain = 0.0
        # for i in range(self.n):
        #     gain+= max(self.similarity_with_nearest_in_effective_x[i], self.sijs[element][i]) - self.similarity_with_nearest_in_effective_x[i]
        # gain += self.query_cap
        # return 
        memo = self.similarity_with_nearest_in_effective_x
        candidate_sim = self.sijs[:, element]
        new_best = torch.maximum(memo, candidate_sim)
        gain = torch.minimum(new_best, self.query_cap) - torch.minimum(memo, self.query_cap)
        return gain.sum().item()


    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not evaluate_set:
            return 0.0
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        cov = torch.sum(self.similarity_with_nearest_in_effective_x).item()
        return cov + self.query_cap[evaluate_set]
    

    def updateMemoization(self , X,element):
        """Update the memoization for the given set"""
        if not self.memoization_initialized or element in X:
            return
        candidate_sim = self.sijs[self._tensor(element , dtype=torch.long)]
        torch.maximum_(self.similarity_with_nearest_in_effective_x, candidate_sim , out=self.similarity_with_nearest_in_effective_x)


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
        return self.effective_ground_set.copy()