#from ..cal_simi_kernel import DenseSimilarity
#from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np


import torch
import torch.nn.functional as F

from abc import ABC , abstractmethod
import torch
from submodlib.runtime import get_default_device


#from .lazier_greedy import LazierThanLazyGreedy
import torch

class NaiveGreedy():
    """
    Naive Greedy optimizer implementation.
    
    At each step, selects the element with maximum marginal gain.
    """

    def __init__(self):
        pass

    def optimize(self, function, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """
        Optimize the submodular function using naive greedy algorithm.
        
        Args:
            function: The submodular function to optimize
            budget: Maximum number of elements to select
            stopIfZeroGain: Stop if marginal gain becomes zero
            stopIfNegativeGain: Stop if marginal gain becomes negative
            epsilon: Not used in naive greedy
            verbose: Print progress information
            show_progress: Show progress bar
            costs: Not used in this implementation
            costSensitiveGreedy: Not used in this implementation
            
        Returns:
            List of selected elements
        """
        if verbose:
            print(f"Starting Naive Greedy optimization with budget {budget}")
        
        selected = torch.zeros(function.n , dtype=torch.bool, device=function.device)
        selected_pairs = []

        iterator = range(budget)
        progress = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                progress = tqdm(range(budget), total=budget, desc="Naive Greedy", leave=False)
                iterator = progress
            except ImportError:
                progress = None

        for iteration in iterator:
            if verbose:
                print(f"Iteration {iteration + 1}/{budget}")
            best_gain , best_idx = function.batchedGain(selected)
            best_idx = int(best_idx.item())
            selected[best_idx] = True
            selected_pairs.append((best_idx , best_gain))
            function.updateBatchMemo(best_idx)
            
        return selected_pairs

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer):
        if optimizer=="NaiveGreedy":
            return NaiveGreedy()
        # elif optimizer==Constant.lazier_greedy:
        #     return LazierThanLazyGreedy()
        else:
            raise Exception("Invalid optimizer. Can be {Constant.naive_greedy}, {Constant.stocastic_greedy},  and 'LazierThanLazyGreedy'.")
class BaseFunction(ABC):

    def __init__(self,n , mode="dense",sijs=None,data=None,num_clusters=None,cluster_label=None,metric="cosine",cluster_labels=None,query_data=None,query_sijs=None , device = None) -> None:
        self.device = torch.device(device) if device else get_default_device()
        self.n = n
        self.mode = mode
        self.metric = metric
        self.data = self._tensor(data)
        self.clusters = None
        self.cluster_sijs = None
        self.cluster_map = None
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters
        self.sijs = sijs
        if query_data is not None:
            self.query_data = self._tensor(query_data)
        self.query_sijs = query_sijs
        
    
    def _tensor(self, val , **kwargs):
        return torch.as_tensor(val , device=self.device, **kwargs)
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.similarity_with_nearest_in_effective_x = self._tensor(torch.zeros(self.n, dtype=torch.float32))
        self.memoization_initialized = True

    @abstractmethod
    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        pass
    @abstractmethod
    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        pass
    @abstractmethod
    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        pass 
    @abstractmethod
    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        pass
    @abstractmethod
    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        pass
    @abstractmethod
    def updateMemoization(self , X):
        """Update the memoization for the given set"""
        pass
    @abstractmethod
    def clearMemoization(self):
        """Clear the memoization"""
        pass
    @abstractmethod
    def setMemoization(self , X):
        """Set the memoization for the given set"""
        pass
    @abstractmethod
    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        pass
class DenseSimilarity:

    @staticmethod
    def euclidean_distance(data, candidate_data , sigma=1.0):
        dist = torch.cdist(data, candidate_data, p=2)
        feature_dim = data.size(-1)
        gamma = 1.0 / feature_dim
        sim = torch.exp(-dist * gamma)
        return sim
    
    @staticmethod
    def cosine_similarity(ground_set, candidate):
        ground_norm = F.normalize(ground_set, p=2, dim=1)
        candidate_norm = F.normalize(candidate, p=2, dim=1)
        return torch.matmul(ground_norm , candidate_norm.T)
    
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
        if self.query_query_sijs is not None:
            """TODO : Validate query_sijs"""
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if isinstance(self.query_data, np.ndarray):
                self.query_data = self._tensor(self.query_data, dtype=torch.float32)
            if self.metric == "euclidean":
                self.query_query_sijs = DenseSimilarity.euclidean_distance(self.query_data , self.query_data)
            elif self.metric == "cosine":
                self.query_query_sijs = DenseSimilarity.cosine_similarity(self.query_data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided")
        if self.query_sijs is not None:
            """TODO : Validate query_sijs"""
        else:
            if self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            if self.metric == "euclidean":
                self.query_sijs = DenseSimilarity.euclidean_distance(self.data , self.query_data)
            elif self.metric == "cosine":
                self.query_sijs = DenseSimilarity.cosine_similarity(self.data , self.query_data)
            else:   
                raise Exception("ERROR: Neither query data matrix nor query similarity kernel provided")
            
        self.similarity_with_nearest_in_effective_x=None
        self.memoization_initialized = False
        self._initialize_memoization()
        self._initialize_ground_sets()
    
    def _initialize_memoization(self):
        self.selected_item = []
        self.P_rows = self._tensor(torch.zeros((0, self.num_queries), dtype=torch.float32))
        self.L_S = None
        self.L_M = None
        self.logdet_S = 0.0
        self.logdet_M = 0.0
        eye_q = self._tensor(torch.eye(self.num_queries, dtype=torch.float32))
        self.query_query_sijs_inv = torch.linalg.inv(self.query_query_sijs + self.eps * eye_q)
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
        eps = 1e-10
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
        P = (self.lambdaVal**2) * (SAQ @ self.query_query_sijs_inv @ SAQ.transpose(1,2))

        eye = torch.eye(k+1 , device=self.device).unsqueeze(0)
        SAA_stable = SAA + eps * eye
        M_stable = SAA - P + eps * eye

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