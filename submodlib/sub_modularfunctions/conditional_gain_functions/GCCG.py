from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np
from submodlib.runtime import get_default_device
from ..regular_submod_functions.graph_cut import GraphCut

class GraphCutConditionalGain:
    
    def __init__(self, n=None, num_privates=None, data_sijs=None,private_sijs=None, data=None,
                 privateData=None, metric="cosine" , privacyHardness=1 , device = None , lambdaVal=None):
        
        if lambdaVal is None:
            raise Exception("ERROR: lambdaVal parameter not provided")
        self.n = n
        self.num_privates = num_privates
        self.lambdaVal = lambdaVal
        self.data_sijs = data_sijs
        self.private_sijs = private_sijs
        self.data = data
        self.privateData = privateData
        self.metric = metric
        self.privacyHardness = privacyHardness
        self.private_sijs=private_sijs
        self.device = torch.device(device) if device else get_default_device()
        self.similarity_with_nearest_in_effective_x = None
        

        if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
        if isinstance(self.data, np.ndarray):
            self.data = self._tensor(self.data, dtype=torch.float32)
        if self.metric == "euclidean":
            self.data_sijs = DenseSimilarity.euclidean_distance(self.data, self.data)
        if self.metric == "cosine":
             self.data_sijs = DenseSimilarity.cosine_similarity(self.data , self.data)
        if self.privateData is None:
            raise Exception("ERROR: Data matrix not provided")
        if isinstance(self.privateData, np.ndarray):
            self.privateData = self._tensor(self.privateData, dtype=torch.float32)
        if self.metric == "euclidean":
             self.private_sijs = DenseSimilarity.euclidean_distance(self.data , self.privateData)
        if self.metric == "cosine":
             self.private_sijs = DenseSimilarity.cosine_similarity(self.data , self.privateData)
        self.private_set = None
        self.private_sum = self.privacyHardness * self.private_sijs.sum(dim=1)
        self.total_similarity_with_master = None
        self.total_similarity_with_subset = None
        self.self_sim = None
        self.graph_cut = None
        self._initialize_ground_sets()
        self._initialize_memoization()
        self.total_similarity_with_master = self.data_sijs.sum(dim=0) 
        self.total_similarity_with_subset = torch.zeros(self.n , device=self.device)
        self.self_sim = self.data_sijs.diagonal().clone()
        self.memoization_initialized = True

    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def _initialize_memoization(self):
        """Initialize memoization structures"""    
        self.total_similarity_with_master = self.data_sijs.sum(dim=0)
        self.total_similarity_with_subset = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.self_sim = torch.diagonal(self.data_sijs).clone()
        self.memoization_initialized = True
    # def initialize_private_set(self):
    #     self.private_set = torch.zeros(self.num_privates , dtype=torch.bool , device=self.device)
    def _tensor(self, val , **kwargs):
        return torch.as_tensor(val , device=self.device, **kwargs)
    
    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)
   

    def batchedGain(self , selected_mask):
        remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
        if remaining.numel()==0:
            return torch.tensor(0.0, device=self.device), torch.tensor(-1, device=self.device)
        
        privacy_term = self.private_sum[remaining]
        master_term = self.total_similarity_with_master[remaining]
        subset_term = self.total_similarity_with_subset[remaining]
        diag_term = self.self_sim[remaining]
        gains = master_term - 2 * self.lambdaVal * subset_term - self.lambdaVal * diag_term - privacy_term
        best_gain , rel_idx = gains.max(dim=0)
        best_idx = remaining[rel_idx]
        return best_gain , best_idx 

    
    def updateBatchMemo(self, element):
        self.total_similarity_with_subset += self.data_sijs[:, element]
    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        return self.effective_ground_set
