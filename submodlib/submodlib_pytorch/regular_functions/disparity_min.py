from submodlib.submodlib_pytorch.base_function import BaseFunction
from submodlib.submodlib_pytorch.cal_simi_kernel import DenseSimilarity
from submodlib.submodlib_pytorch.optimizers.optimizer_factory import OptimizerFactory
from submodlib.submodlib_pytorch.userValidator import validate_mode, validate_n, validate_sijs

import numpy as np
import torch


class DisparityMin(BaseFunction):
    """
    DisparityMin objective:
        f(X) = min_{i,j in X, i != j} (1 - s_ij)
    """

    def __init__(self, n, mode="dense", sijs=None, data=None, metric="cosine", device="cpu"):
        super().__init__(n=n, mode=mode, sijs=sijs, data=data, metric=metric, device=device)

        validate_n(self.n)
        validate_mode(self.mode)
        if self.mode != "dense":
            raise Exception("ERROR: DisparityMin currently supports only dense mode")

        if self.sijs is not None:
            if isinstance(self.sijs, np.ndarray):
                self.sijs = self._tensor(self.sijs, dtype=torch.float32)
            else:
                self.sijs = self._tensor(self.sijs, dtype=torch.float32)
            validate_sijs(type(self.sijs), self.mode)
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            if isinstance(self.data, np.ndarray):
                self.data = self._tensor(self.data, dtype=torch.float32)
            if self.metric == "euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data, self.data)
            elif self.metric == "cosine":
                # Normalize to [0, 1] because DisparityMin expects normalized similarities.
                self.sijs = DenseSimilarity.cosine_similarity(self.data, self.data)
                self.sijs = (self.sijs + 1.0) * 0.5
            else:
                raise Exception("ERROR: Unsupported metric")

        self.sijs = torch.clamp(self.sijs, 0.0, 1.0)
        self.dijs = 1.0 - self.sijs
        self._initialize_ground_sets()
        self._initialize_memoization()

    def _initialize_ground_sets(self):
        # Match C++ semantics where effective ground set is an unordered set of indices.
        self.effective_ground_set = set(range(self.n))

    def _initialize_memoization(self):
        self.selected_count = 0
        self.current_min_distance = self._tensor(0.0, dtype=torch.float32)
        self.min_dist_to_selected = self._tensor(torch.full((self.n,), float("inf")), dtype=torch.float32)
        self.memoization_initialized = True

    def maximize(
        self,
        optimizer="NaiveGreedy",
        budget=None,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=0.1,
        verbose=False,
        show_progress=True,
        costs=None,
        costSensitiveGreedy=False,
    ):
        if budget is None or budget <= 0:
            raise Exception("Budget cannot be zero or negative")
        if optimizer != "NaiveGreedy":
            raise Exception("ERROR: DisparityMin currently supports only NaiveGreedy in PyTorch")
        if costs is not None and len(costs) != 0:
            raise Exception("ERROR: costs/costSensitiveGreedy path is not implemented for DisparityMin PyTorch")
        if costSensitiveGreedy:
            raise Exception("ERROR: costSensitiveGreedy is not implemented for DisparityMin PyTorch")

        # Deterministic vectorized greedy:
        # - gains are computed for all candidates in batched form
        # - ties are resolved by torch.argmax first-occurrence behavior
        #   on ascending index order (via remaining from nonzero(mask)).
        
        greedy_vector = []
        selected_mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.clearMemoization()

        for _ in range(min(int(budget), self.n)):
            best_gain, best_idx = self.batchedGain(selected_mask)
            if int(best_idx.item()) == -1:
                break
            if stopIfNegativeGain and best_gain.item() < 0:
                break
            if stopIfZeroGain and best_gain.item() == 0:
                break

            best_id = int(best_idx.item())
            selected_mask[best_idx] = True
            self.updateBatchMemo(best_id)
            greedy_vector.append((best_id, float(best_gain.item())))

        return greedy_vector

    def batchedGain(self, selected_mask):
        remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(-1, device=self.device)

        if self.selected_count == 0:
            gains = torch.zeros_like(remaining, dtype=torch.float32, device=self.device)
        elif self.selected_count == 1:
            gains = self.min_dist_to_selected[remaining]
        else:
            candidate_new_obj = torch.minimum(
                self.min_dist_to_selected[remaining],
                self.current_min_distance.expand(remaining.numel()),
            )
            gains = candidate_new_obj - self.current_min_distance

        best_gain, rel_idx = gains.max(dim=0)
        best_idx = remaining[rel_idx]
        return best_gain, best_idx

    def evaluate(self, evaluate_set):
        if len(evaluate_set) < 2:
            return 0.0

        X_tensor = self._tensor(torch.tensor(list(evaluate_set), dtype=torch.long))
        dist_sub = self.dijs[X_tensor][:, X_tensor].clone()
        dist_sub.fill_diagonal_(float("inf"))
        return torch.min(dist_sub).item()

    def marginalGain(self, X, element):
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        current_val = self.evaluate(X)
        new_val = self.evaluate(X | {element})
        return new_val - current_val

    def marginalGainWithMemoization(self, X, element):
        if element in X:
            return 0.0
        if element not in self.effective_ground_set:
            return 0.0
        if not self.memoization_initialized:
            return self.marginalGain(X, element)

        if self.selected_count == 0:
            return 0.0
        if self.selected_count == 1:
            return self.min_dist_to_selected[element].item()

        new_obj = min(self.current_min_distance.item(), self.min_dist_to_selected[element].item())
        return new_obj - self.current_min_distance.item()

    def evaluateWithMemoization(self, evaluate_set):
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        if self.selected_count < 2:
            return 0.0
        return self.current_min_distance.item()

    def updateMemoization(self, X, element):
        if element in X:
            return
        self.updateBatchMemo(element)

    def updateBatchMemo(self, element):
        if self.selected_count == 0:
            self.min_dist_to_selected = self.dijs[:, element].clone()
            self.current_min_distance = self._tensor(0.0, dtype=torch.float32)
            self.selected_count = 1
            return

        min_to_current_set = self.min_dist_to_selected[element]
        if self.selected_count == 1:
            self.current_min_distance = min_to_current_set
        else:
            self.current_min_distance = torch.minimum(self.current_min_distance, min_to_current_set)

        self.min_dist_to_selected = torch.minimum(self.min_dist_to_selected, self.dijs[:, element])
        self.selected_count += 1

    def clearMemoization(self):
        self.selected_count = 0
        self.current_min_distance = self._tensor(0.0, dtype=torch.float32)
        self.min_dist_to_selected.fill_(float("inf"))

    def setMemoization(self, X):
        self.clearMemoization()
        for ele in X:
            self.updateBatchMemo(ele)

    def getEffectiveGroundSet(self):
        return set(self.effective_ground_set)
