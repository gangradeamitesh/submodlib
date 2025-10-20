from submodlib.sub_modularfunctions.base_function import BaseFunction
import userValidator 
from userValidator import validate_n, validate_sep_rep , validate_sijs
import torch
from cal_simi_kernel import DenseSimilarity
import numpy as np

class FacilityLocationMutualInfoFunction(BaseFunction):
    def __init__(self, n , num_queries , data_sijs=None, query_sijs=None,
                 data=None, query_data=None, metric="cosine", magnificationEta=1):
        """
        Initializes the Facility Location Mutual Information Function.

        Parameters:
        - data: The ground set data points.
        - query_data: The query set data points.
        - num_neighbors: Number of neighbors to consider for mutual information calculation.
        """

        super().__init__(n=n, sijs=data_sijs, data=data, metric=metric,query_data=query_data,query_sijs=query_sijs)
        self.magnificationEta = magnificationEta
        self.effective_ground_set = None

        validate_n(self.n)
        if self.sijs is not None:
            """TODO: Validate data_sijs"""
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = torch.tensor(self.data, dtype=torch.float32)
            
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
                self.query_data = torch.tensor(self.query_data, dtype=torch.float32)
            if self.metric == "euclidean":
                self.query_sijs - DenseSimilarity.euclidean_distance(self.data , self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided") 

        self._initialize_ground_sets()
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
            # Create ground set with items 0 to n-1 (like C++ lines 28-32)
        self.effective_ground_set = set(range(self.n))

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        pass

    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        pass

    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
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
        