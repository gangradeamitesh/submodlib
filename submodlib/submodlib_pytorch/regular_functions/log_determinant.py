from pickle import NONE
from submodlib.submodlib_pytorch.base_function import BaseFunction
from submodlib.submodlib_pytorch.optimizers.optimizer_factory import OptimizerFactory
import torch
from submodlib.submodlib_pytorch.userValidator import validate_n, validate_mode, validate_sep_rep , validate_sijs
from submodlib.submodlib_pytorch.cal_simi_kernel import DenseSimilarity
import numpy as np

class LogDeterminant(BaseFunction):
    
    def __init__(self, n, mode="dense", sijs=None, 
                 data=None, 
                 metric="cosine",device="cpu",lambdaVal=1.0) -> None:
        if data is None and sijs is None:
            raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        validate_n(n)
        validate_mode(mode)

        super().__init__(n=n,mode= mode,sijs= sijs,data= data,metric= metric,device=device)
        self.optimizer = None
        
        self.memoization_initialized = False
        self.lambdaVal = lambdaVal
        
        self.effective_ground_set = None
        self._initialize_ground_sets()
        
        #validate_sep_rep(self.separate_rep, self.mode, self.n_rep)
        if self.sijs is not None:
            validate_sijs(type(self.sijs))
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)
            
            if self.mode == "dense" and self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data,self.data)
            elif self.mode == "dense" and self.metric == "cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data,self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
    
    def _initialize_ground_sets(self):
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget=budget , stopIfZeroGain=stopIfZeroGain , stopIfNegativeGain=stopIfNegativeGain , epsilon=epsilon , verbose=verbose , show_progress=show_progress)
    
    def batchedGain(self, selected_mask):
        selected = selected_mask.nonzero(as_tuple=False).flatten()
        remaining = (~selected_mask).nonzero(as_tuple=False).flatten()

        if remaining.numel() == 0:
            return (
                torch.tensor(float("-inf"), device=self.device),
                torch.tensor(-1, device=self.device),
            )

        if selected.numel() == 0:
            current_val = torch.tensor(0.0, device=self.device, dtype=self.sijs.dtype)
        else:
            L_X = self.sijs.index_select(0, selected).index_select(1, selected)
            eye = torch.eye(L_X.size(0), device=self.device, dtype=L_X.dtype)
            L_X = L_X + self.lambdaVal * eye
            sign_x, logdet_x = torch.linalg.slogdet(L_X)
            current_val = (
                logdet_x
                if sign_x > 0
                else torch.tensor(float("-inf"), device=self.device, dtype=self.sijs.dtype)
            )

        candidate_indices = torch.cat(
            [
                selected.unsqueeze(0).expand(remaining.size(0), -1),
                remaining.unsqueeze(1),
            ],
            dim=1,
        )  # [num_remaining, |X| + 1]

        submats = self.sijs[candidate_indices.unsqueeze(2), candidate_indices.unsqueeze(1)]
        # [num_remaining, |X| + 1, |X| + 1]

        eye = torch.eye(
            submats.size(-1),
            device=self.device,
            dtype=submats.dtype,
        ).unsqueeze(0)

        submats = submats + self.lambdaVal * eye

        signs, logdets = torch.linalg.slogdet(submats)

        candidate_vals = torch.where(
            signs > 0,
            logdets,
            torch.full_like(logdets, float("-inf")),
        )

        gains = candidate_vals - current_val

        best_pos = torch.argmax(gains)
        best_gain = gains[best_pos]
        best_idx = remaining[best_pos]

        return best_gain, best_idx
    
    def updateBatchMemo(self, element):
        pass


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
        if not evaluate_set:
            return 0.0
        X_list = list(evaluate_set)
        X_tensor = torch.tensor(X_list, dtype=torch.long)
        # effective_ground_tensor = torch.tensor(list(self.effective_ground_set), dtype=torch.long)
        det_X_tensor = self.sijs[X_tensor][:, X_tensor]
        return torch.logdet(det_X_tensor)

    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        gain = 0.0
        element_tensor = self._tensor([element], dtype=torch.long)
        new_similarities = self.sijs[:, element_tensor]
        gain_tensor = torch.maximum(new_similarities - self.similarity_with_nearest_in_effective_x, torch.tensor(0.0))
        gain = torch.sum(gain_tensor).item()
        return gain

    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        
        return torch.sum(self.similarity_with_nearest_in_effective_x).item()


    def updateMemoization(self , X,element):
        """Update the memoization for the given set"""
        if not self.memoization_initialized or element in X:
            return
        element_tensor = self._tensor(element, dtype=torch.long)
        new_similarity = self.sijs[:, element_tensor]
        torch.maximum(new_similarity, self.similarity_with_nearest_in_effective_x, out=self.similarity_with_nearest_in_effective_x)


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