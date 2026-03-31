from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np

class FacilityLocationVariantMutualInformation(BaseFunction):
    def __init__(self, n , num_queries , query_sijs=None,
                 data=None, queryData=None, metric="cosine", queryDiversityEta=1,device=None):
        """
        Initializes the Facility Location Mutual Information Function.

        Parameters:
        - data: The ground set data points.
        - query_data: The query set data points.
        - num_neighbors: Number of neighbors to consider for mutual information calculation.
        """

        super().__init__(n=n, data=data, metric=metric,query_data=queryData,query_sijs=query_sijs , device=device)
        self.queryDiversityEta = queryDiversityEta
        self.effective_ground_set = None
        self.num_queries = num_queries



        validate_n(self.n)
        # if self.sijs is not None:
        #     """TODO: Validate data_sijs"""
        # else:
        #     if self.data is None:
        #         raise Exception("ERROR: Data matrix not provided")
            
        #     if isinstance(self.data, np.ndarray):
        #         self.data = self._tensor(self.data, dtype=torch.float32)
            
        #     if self.metric == "euclidean":
        #         self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
        #     elif self.metric == "cosine":
        #         self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
        #     else:
        #         raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
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
            elif self.metric == "rbf":
                self.query_sijs = DenseSimilarity.rbf_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided") 
        self.effective_query_set = None
        self._initialize_ground_sets()
        self._initialize_query_sets()
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()
        self.query_cap = torch.max(self.query_sijs, dim=1).values
    
    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.num_queries, dtype=torch.float32))
        self.memoization_initialized = True

    def _initialize_query_sets(self):
        self.effective_query_set = self._tensor(torch.arange(self.num_queries), dtype=torch.long)

    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon=False , verbose=False , show_progress=False , costs=None , costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress)

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
        gain = 0.0
        if len(evaluate_set) == 0:
            return 0
        ids = self._tensor(list(evaluate_set), dtype=torch.long)
        per_query_set = self.query_sijs[ids].max(dim=0).values
        first_term = torch.sum(per_query_set)
        second_term = self.queryDiversityEta * torch.sum(self.query_cap[ids])
        return first_term + second_term

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        
        # idx = int(element)
        # if idx in X:
        #     return 0.0

        # candidate = self.query_sijs[idx] 
        # new_best = torch.maximum(self.similarity_with_nearest_in_effective_x, candidate)
        # delta_queries = torch.sum(new_best - self.similarity_with_nearest_in_effective_x)
        # delta_items = self.queryDiversityEta * self.query_cap[idx]
        # return delta_queries + delta_items
        
    
    def batchedGain(self ,X):
        gain = torch.maximum(self.similarity_with_nearest_in_effective_x , self.query_sijs) - self.similarity_with_nearest_in_effective_x
        gain = gain.sum(dim=1) + self.queryDiversityEta * self.query_cap
        gain = gain.masked_fill(X, float("-inf"))
        return gain.max(dim=0)

    def updateBatchMemo(self,element):
        candidate = self.query_sijs[element]
        self.similarity_with_nearest_in_effective_x = torch.maximum(self.similarity_with_nearest_in_effective_x, candidate)

    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        res= 0.0
        if len(evaluate_set) == 0:
            return 0
        # for i in range(self.num_queries):
        #     res += self.similarity_with_nearest_in_effective_x[i].item()

        res = self.similarity_with_nearest_in_effective_x.sum()
        res += self.queryDiversityEta * torch.sum(self.query_cap[self._tensor(list(evaluate_set), dtype=torch.long)])
        return res
    

    def updateMemoization(self , X,element):
        """Update the memoization for the given set"""
        idx = int(element)
        if idx in X:
            return
        candidate = self.query_sijs[idx]
        torch.maximum(self.similarity_with_nearest_in_effective_x, candidate, out=self.similarity_with_nearest_in_effective_x)


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