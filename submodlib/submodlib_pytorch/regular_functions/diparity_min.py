from submodlib.submodlib_pytorch.userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
import torch
import numpy as np
from submodlib.submodlib_pytorch.cal_simi_kernel import DenseSimilarity
from submodlib.submodlib_pytorch.base_function import BaseFunction
from submodlib.submodlib_pytorch.optimizers.optimizer_factory import OptimizerFactory

class DiparityMin(BaseFunction):
    def __init__(self, n, mode="dense", seperate_rep=None, n_rep=None, sijs=None, 
                 data=None, data_rep=None, num_clusters=None, cluster_labels=None, 
                 metric="cosine", num_neighbors=None, create_dense_cpp_kernel_in_python=True, 
                 pybind_mode=None, partial=False, ground_set=None, separate_master=False):
        
        super().__init__(n=n, mode=mode, sijs=sijs, data=data,cluster_label=cluster_labels , num_clusters=num_clusters, metric=metric)

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
        
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        
        # Effective ground set - this is what getEffectiveGroundSet() should return
        self.effective_ground_set = None
        self.master_set = None
        self.n_master = None
        
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
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif self.create_dense_kernel == True and self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        
        self._initialize_ground_sets()
        
        # Initialize memoization
        """TODO: Commented for now, to be revisited later"""
        # self._initialize_memoization()

    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        if self.partial and self.ground_set is not None:
            # Use provided ground set (partial mode)
            self.effective_ground_set = set(self.ground_set)
        else:
            # Create ground set with items 0 to n-1 (like C++ lines 28-32)
            self.effective_ground_set = set(range(self.n))
        
        # Determine master set
        if self.separate_master:
            # Master set is separate from ground set
            if self.sijs is not None:
                self.n_master = self.sijs.shape[0]
                self.master_set = set(range(self.n_master))
            else:
                raise Exception("ERROR: separate_master=True requires similarity kernel")
        else:
            # Master set is same as effective ground set (like C++ lines 88-90)
            self.n_master = len(self.effective_ground_set)
            self.master_set = self.effective_ground_set.copy()

    def _initialize_memoization(self):
        """Initialize memoization structures"""
        if self.sijs is not None and self.n_master is not None:
            self.similarity_with_nearest_in_effective_x = torch.zeros(self.n_master, dtype=torch.float32)
            self.memoization_initialized = True

    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_factory = OptimizerFactory()
        optimizer_obj = optimizer_factory.get_optimizer(optimizer)
        output = optimizer_obj.optimize(self, budget, stopIfZeroGain, stopIfNegativeGain, 
                                       epsilon, verbose, show_progress, costs, costSensitiveGreedy)
        return output

    def evaluate(self, evaluate_set):
        if not evaluate_set:
            return 0.0
        

        if self.partial:
            effective_x = evaluate_set & self.effective_ground_set
        else:
            effective_x = evaluate_set
        
        if not effective_x:
            return 0.0
        X_tensor = torch.tensor(list(effective_x), dtype=torch.long)
        sijs_submatrix = 1 - self.sijs[X_tensor][:, X_tensor]
        sijs_submatrix.fill_diagonal_(float('inf'))
        return torch.min(sijs_submatrix).item()
        

    def marginalGain(self, X, element):
        """
        Compute marginal gain of adding element to set X.
        This is the difference in function value when element is added.
        """
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        current_val = self.evaluate(X)
        X_new = X | {element}
        new_val = self.evaluate(X_new)
        return new_val - current_val

    def marginalGainWithMemoization(self, X, element):
        """
        Compute marginal gain using memoization for efficiency.
        This is the key optimization that makes greedy algorithms fast.
        """
        pass

    def evaluateWithMemoization(self, evaluate_set):
        """
        Evaluate using pre-computed memoization.
        Assumes memoization is up-to-date for the given set.
        """
        pass

    def updateMemoization(self, X, element):
        """
        Update memoization after adding element to set X.
        This is called after each greedy selection.
        """
        pass

    def clearMemoization(self):
        """Clear all memoization data"""
        pass

    def setMemoization(self, X):
        """
        Set memoization for a given set X.
        This initializes memoization as if X was the current set.
        """
        pass

    def getEffectiveGroundSet(self):
        """
        Return the effective ground set (actual set, not size).
        This matches the C++ implementation.
        """
        return self.effective_ground_set.copy()  # Return a copy to prevent external modification