from pickle import NONE
from submodlib.sub_modularfunctions.base_function import BaseFunction
from submodlib.sub_modularfunctions.optimizers.optimizer_factory import OptimizerFactory
import torch
from submodlib.sub_modularfunctions.userValidator import validate_n, validate_mode, validate_sep_rep , validate_sijs
from submodlib.sub_modularfunctions.cal_simi_kernel import DenseSimilarity

class LogDeterminant(BaseFunction):
    
    def __init__(self, n, mode="dense", seperate_rep=None, n_rep=None, sijs=None, 
                 data=None, data_rep=None, num_clusters=None, cluster_labels=None, 
                 metric="cosine", num_neighbors=None, create_dense_cpp_kernel_in_python=True, 
                 pybind_mode=None, partial=False, ground_set=None, separate_master=False ) -> None:
        super().__init__(n, mode, sijs, data, num_clusters, metric, cluster_labels)
        self.n_rep = n_rep
        self.data_rep = data_rep
        self.num_neighbors = num_neighbors
        self.separate_rep = seperate_rep
        self.effective_ground = None
        self.create_dense_kernel = create_dense_cpp_kernel_in_python
        self.optimizer = None
        
        # New parameters for proper ground set handling
        self.partial = partial
        self.ground_set = ground_set
        self.separate_master = separate_master
        
        # Memoization variables (similar to C++ version)
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        
        # Effective ground set - this is what getEffectiveGroundSet() should return
        self.effective_ground_set = None
        self.master_set = None
        self.n_master = None
        self._initialize_ground_sets()
        validate_n(self.n)
        validate_mode(self.mode)
        validate_sep_rep(self.separate_rep, self.mode, self.n_rep)
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
            
            if self.create_dense_kernel == True and self.mode == "dense" and self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data)
            elif self.create_dense_kernel == True and self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        if self.partial and self.ground_set is not None:
            self.effective_ground_set = set(self.ground_set)
        else:
            self.effective_ground_set = set(range(self.n))

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
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
        if self.partial:
            effective_x = evaluate_set & self.effective_ground_set
        else:
            effective_x = evaluate_set
        if not effective_x:
            return 0.0
        X_list = list(effective_x)
        X_tensor = torch.tensor(X_list, dtype=torch.long)
        # effective_ground_tensor = torch.tensor(list(self.effective_ground_set), dtype=torch.long)
        det_X_tensor = self.sijs[X_tensor][:, X_tensor]
        return torch.logdet(det_X_tensor)

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
    