from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np
from ..userValidator import validate_sijs

import torch
import torch.nn.functional as F

from abc import ABC , abstractmethod
import torch
from submodlib.runtime import get_default_device

import torch    
class LogDeterminantMutualInformation(BaseFunction):

    def __init__(self, n ,num_queries,lambdaVal=1.0, data_sijs=None, query_sijs=None, query_query_sijs=None, data=None, queryData=None, metric="cosine",magnificationEta=1):
        
        super().__init__(
    n=n,
    sijs=data_sijs,
    data=data,
    metric=metric,
    query_data=queryData,
    query_sijs=query_sijs,
           
    )

        self.lambdaVal = lambdaVal
        self.magnificationEta = magnificationEta
        self.query_query_sijs = query_query_sijs
        self.eps = 1e-10
        self.num_queries = self.query_data.shape[0] if self.query_data is not None else 0

 
        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")

        # if self.num_queries < 0:
        #     raise Exception("ERROR: Number of queries must be >= 0")
#        validate_n(self.n)
        if self.sijs is not None:
            self.sijs = self._tensor(self.sijs)
            validate_sijs(type(self.sijs))
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float16)
            
            if self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data, self.data)
            elif self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data, self.data)
            elif self.metric == "rbf":
                self.sijs = DenseSimilarity.rbf_similarity(self.data , self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        if self.query_query_sijs is not None:
            self.query_query_sijs = self._tensor(self.query_query_sijs)
            validate_sijs(type(self.query_query_sijs))
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if isinstance(self.query_data, np.ndarray):
                self.query_data = self._tensor(self.query_data, dtype=torch.float16)
            if self.metric == "euclidean":
                self.query_query_sijs = DenseSimilarity.euclidean_distance(self.query_data , self.query_data)
            elif self.metric == "cosine":
                self.query_query_sijs = DenseSimilarity.cosine_similarity(self.query_data , self.query_data)
            elif self.metric == "rbf":
                self.query_query_sijs = DenseSimilarity.rbf_similarity(self.query_data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided")
        if self.query_sijs is not None:
            self.query_sijs = self._tensor(self.query_sijs)
            validate_sijs(type(self.query_sijs))
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if self.metric == "euclidean":
                self.query_sijs = DenseSimilarity.euclidean_distance(self.data , self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
            elif self.metric == "rbf":
                self.query_sijs = DenseSimilarity.rbf_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided")
            
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()
        self._initialize_ground_sets()
    
    def _initialize_memoization(self):
        self.selected_item = []
        self.P_rows = self._tensor(torch.zeros((0, self.num_queries), dtype=torch.float16))
        self.L_S = None
        self.L_M = None
        self.logdet_S = 0.0
        self.logdet_M = 0.0
        #eye_q = self._tensor(torch.eye(self.num_queries, dtype=torch.float16))
        self.query_query_sijs_inv = torch.linalg.inv(self.query_query_sijs)
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
    
    def batchedGain(self ,X):
        #eps = 1e-10
        remaining = (~X).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            return torch.tensor(0., device=self.device), torch.tensor(-1, device=self.device)
        
        seleced_idx = X.nonzero(as_tuple=False).flatten()
        k = seleced_idx.numel()
        idx_stack = torch.zeros((remaining.numel(), k + 1), dtype=torch.long, device=self.device)
        if k > 0:
            idx_stack[:, :k] = seleced_idx
        
        idx_stack[:, k] = remaining
        rows = idx_stack.unsqueeze(2)
        cols = idx_stack.unsqueeze(1)
        SAA = self.sijs[rows, cols]
        #SAA = self.sijs[idx_stack][:, idx_stack.transpose(1,2)]
        SAQ = self.query_sijs[idx_stack]
        P = (self.magnificationEta**2) * (SAQ @ self.query_query_sijs_inv @ SAQ.transpose(1,2))

        eye = torch.eye(k+1 , device=self.device).unsqueeze(0)
        # SAA_stable = SAA + eps * eye
        # M_stable = SAA - P + eps * eye
        SAA_stable = SAA
        M_stable = SAA - P

        logdet_S = torch.logdet(SAA_stable)
        logdet_M = torch.logdet(M_stable)
        gains = logdet_S - logdet_M

        best_gain , best_idx = gains.max(dim=0)
        return best_gain , remaining[best_idx]


    def updateBatchMemo(self, element):
    # For the direct logdet approach, memo can just track the set.
        if not hasattr(self, "selected_items"):
            self.selected_items = []
        self.selected_items.append(int(element))  

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        return self.sijs[element].sum().item()

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