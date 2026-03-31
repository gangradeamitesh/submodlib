
from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np

"""TODO : To implement all the functionlity for Graph Cut Mutual Information Function"""

class GraphCutMutualInformation(BaseFunction):
    def __init__(self, n , num_queries, query_sijs=None,
                 data=None, queryData=None, metric="cosine"):
        """
        Initializes the Facility Location Mutual Information Function.

        Parameters:
        - data: The ground set data points.
        - query_data: The query set data points.
        - num_neighbors: Number of neighbors to consider for mutual information calculation.
        """

        super().__init__(n=n, data=data, metric=metric,query_data=queryData,query_sijs=query_sijs)
        self.evalX = 0.0

        validate_n(self.n)
        if self.sijs is not None:
            """TODO: Validate data_sijs"""
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)

            
            if self.metric == "euclidean":
                self.query_sijs = DenseSimilarity.euclidean_distance(self.data,self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data,self.query_data)
            elif self.metric == "rbf":
                self.query_sijs = DenseSimilarity.rbf_similarity(self.data , self.query_data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        # if self.query_sijs is not None:
        #     """TODO : Validate query_sijs"""
        # else:
        #     if self.query_data is None:
        #         raise Exception("ERROR: Query data matrix not provided")
        #     if isinstance(self.query_data, np.ndarray):
        #         self.query_data = torch.tensor(self.query_data, dtype=torch.float32)
        #     if self.metric == "euclidean":
        #         self.query_sijs = DenseSimilarity.euclidean_distance(self.data , self.query_data)
        #     elif self.metric == "cosine":
        #         self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
        #     else:   
        #         raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided") 
                self._initialize_ground_sets()
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()
        self._initialize_ground_sets()
        self.query_sum = self.query_sijs.sum(dim=1)
    
    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.query_data.shape[0], dtype=torch.float32))
        self.memoization_initialized = True
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = set(range(self.n))

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)

    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        #return torch.sum(self.sijs[element , :]) * 2
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        return torch.sum(self.sijs[element]).item()

    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        if not evaluate_set:
            return 0.0
        return self.sijs[self._tensor(list(evaluate_set), dtype=torch.long) ].sum().item()

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        return self.sijs[element].sum().item()
    
    """Gain values do not match exactly due to difference in implementation of max function in C++ and PyTorch."""
    def batchedGain(self, X):
        gain = self.query_sum.masked_fill(X , float("-inf"))
        return gain.max(dim=0)
    
    def updateBatchMemo(self , element):
        # candidate = self.query_sijs[element]
        # # print("Candidate:")
        # # print(candidate)
        # self.similarity_with_nearest_in_effective_x =  candidate
        # #print(self.similarity_with_nearest_in_effective_x)
        pass

    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not evaluate_set:
            return 0.0
        return self.evalX

    def updateMemoization(self , X,element):
        """Update the memoization for the given set"""
        if element in X:
            return 0.0
        self.evalX += torch.sum(self.sijs[element]).item()

    def clearMemoization(self):
        """Clear the memoization"""
        self.evalX = 0.0

    def setMemoization(self , X):
        """Set the memoization for the given set"""
        self.evalX = self.evaluate(X)

    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        return self.effective_ground_set.copy()