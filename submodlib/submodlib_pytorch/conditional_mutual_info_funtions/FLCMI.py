from ..userValidator import validate_n, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch


class FacilityLocationConditionalMutualInformation(BaseFunction):
    def __init__(
        self,
        n,
        num_queries,
        num_privates,
        data_sijs=None,
        query_sijs=None,
        private_sijs=None,
        data=None,
        queryData=None,
        privateData=None,
        metric="cosine",
        magnificationEta=1,
        privacyHardness=1,
        device=None,
    ):
        """
        Initialize the Facility Location Conditional Mutual Information function.

        This implements the dense-kernel FLCMI objective:
            sum_i max(min(max_{j in A} s_ij, eta max_{j in Q} s_ij)
                      - nu max_{j in P} s_ij, 0)
        """

        super().__init__(
            n=n,
            sijs=data_sijs,
            data=data,
            metric=metric,
            query_data=queryData,
            query_sijs=query_sijs,
            device=device,
        )
        self.num_queries = num_queries
        self.num_privates = num_privates
        self.magnificationEta = magnificationEta
        self.privacyHardness = privacyHardness
        self.privateData = privateData
        self.private_sijs = private_sijs
        self.data_sijs = self.sijs

        validate_n(self.n)
        if self.num_queries < 0:
            raise Exception("ERROR: Number of queries must be >= 0")
        if self.num_privates < 0:
            raise Exception("ERROR: Number of private data points must be >= 0")
        if self.data is not None:
            self.data = self._tensor(self.data, dtype=torch.float32)

        self._initialize_data_kernel()
        self._initialize_query_kernel()
        self._initialize_private_kernel()
        self._initialize_ground_sets()
        self._initialize_query_cap()
        self._initialize_private_penalty()

        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        self._initialize_memoization()

    def _initialize_data_kernel(self):
        if self.data_sijs is not None:
            self.data_sijs = self._tensor(self.data_sijs, dtype=torch.float32)
            validate_sijs(type(self.data_sijs))
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            self.data = self._tensor(self.data, dtype=torch.float32)
            self.data_sijs = self._compute_similarity(self.data, self.data)

        self._validate_kernel_shape("data_sijs", self.data_sijs, (self.n, self.n))
        self.sijs = self.data_sijs

    def _initialize_query_kernel(self):
        if self.query_sijs is not None:
            self.query_sijs = self._tensor(self.query_sijs, dtype=torch.float32)
            validate_sijs(type(self.query_sijs))
        elif self.num_queries == 0:
            self.query_sijs = torch.empty((self.n, 0), dtype=torch.float32, device=self.device)
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            if not hasattr(self, "query_data") or self.query_data is None:
                raise Exception("ERROR: Query data matrix not provided")
            self.query_data = self._tensor(self.query_data, dtype=torch.float32)
            self.query_sijs = self._compute_similarity(self.data, self.query_data)

        self._validate_kernel_shape("query_sijs", self.query_sijs, (self.n, self.num_queries))

    def _initialize_private_kernel(self):
        if self.private_sijs is not None:
            self.private_sijs = self._tensor(self.private_sijs, dtype=torch.float32)
            validate_sijs(type(self.private_sijs))
        elif self.num_privates == 0:
            self.private_sijs = torch.empty((self.n, 0), dtype=torch.float32, device=self.device)
        else:
            if self.data is None:
                raise Exception("ERROR: Data matrix not provided")
            if self.privateData is None:
                raise Exception("ERROR: Private data matrix not provided")
            self.privateData = self._tensor(self.privateData, dtype=torch.float32)
            self.private_sijs = self._compute_similarity(self.data, self.privateData)

        self._validate_kernel_shape("private_sijs", self.private_sijs, (self.n, self.num_privates))

    def _compute_similarity(self, data, candidate_data):
        if self.metric == "euclidean":
            return DenseSimilarity.euclidean_distance(data, candidate_data)
        if self.metric == "cosine":
            return DenseSimilarity.cosine_similarity(data, candidate_data)
        if self.metric == "rbf":
            return DenseSimilarity.rbf_similarity(data, candidate_data)
        raise Exception(f"ERROR: Similarity Metric {self.metric} not recognized.")

    def _validate_kernel_shape(self, name, kernel, expected_shape):
        if kernel.dim() != 2 or tuple(kernel.shape) != expected_shape:
            raise Exception(f"ERROR: {name} should be {expected_shape[0]} X {expected_shape[1]}")

    def _initialize_ground_sets(self):
        """Initialize effective ground set like the C++ wrapper."""
        self.effective_ground_set = self._tensor(torch.arange(self.n), dtype=torch.long)

    def _initialize_query_cap(self):
        if self.num_queries == 0:
            self.query_cap = torch.zeros(self.n, dtype=torch.float32, device=self.device)
            return
        scaled_query = self.magnificationEta * self.query_sijs
        self.query_cap = torch.clamp(scaled_query, min=0).max(dim=1).values

    def _initialize_private_penalty(self):
        if self.num_privates == 0:
            self.private_penalty = torch.zeros(self.n, dtype=torch.float32, device=self.device)
            return
        scaled_private = self.privacyHardness * self.private_sijs
        self.private_penalty = torch.clamp(scaled_private, min=0).max(dim=1).values

    def _initialize_memoization(self):
        """Initialize memoization structures."""
        self.similarity_with_nearest_in_effective_x = torch.zeros(
            self.n, dtype=torch.float32, device=self.device
        )
        self.memoization_initialized = True

    def _normalize_element(self, element):
        if isinstance(element, torch.Tensor):
            if element.numel() != 1:
                raise Exception("ERROR: element must be a scalar index")
            return int(element.detach().item())
        return int(element)

    def _valid_element(self, element):
        return 0 <= element < self.n

    def _contains(self, X, element):
        if X is None:
            return False
        if isinstance(X, torch.Tensor):
            if X.dtype == torch.bool:
                mask = X.to(device=self.device).flatten()
                return self._valid_element(element) and bool(mask[element].detach().item())
            return bool(torch.any(X.to(device=self.device) == element).detach().item())
        return element in X

    def _is_empty_selection(self, X):
        if X is None:
            return True
        if isinstance(X, torch.Tensor):
            if X.dtype == torch.bool:
                return not bool(X.to(device=self.device).any().detach().item())
            return X.numel() == 0
        return len(X) == 0

    def _indices_from_set(self, X):
        if X is None:
            return torch.empty(0, dtype=torch.long, device=self.device)

        if isinstance(X, torch.Tensor):
            if X.dtype == torch.bool:
                if X.numel() != self.n:
                    raise Exception(f"ERROR: Boolean selected mask should have length {self.n}")
                indices = X.to(device=self.device).flatten().nonzero(as_tuple=False).flatten()
            else:
                indices = X.to(device=self.device, dtype=torch.long).flatten()
        else:
            values = list(X)
            if len(values) == 0:
                return torch.empty(0, dtype=torch.long, device=self.device)
            indices = self._tensor(values, dtype=torch.long).flatten()

        if indices.numel() == 0:
            return indices
        if torch.any((indices < 0) | (indices >= self.n)):
            raise Exception(f"ERROR: Selected indices should be in [0, {self.n})")
        return torch.unique(indices)

    def _selected_max(self, X):
        indices = self._indices_from_set(X)
        if indices.numel() == 0:
            return torch.zeros(self.n, dtype=torch.float32, device=self.device)
        selected_sims = torch.clamp(self.data_sijs[:, indices], min=0)
        return selected_sims.max(dim=1).values

    def _score_from_memo(self, memo):
        capped = torch.minimum(memo, self.query_cap)
        return torch.clamp(capped - self.private_penalty, min=0).sum()

    def maximize(
        self,
        optimizer,
        budget,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        epsilon=None,
        verbose=False,
        show_progress=True,
        costs=None,
        costSensitiveGreedy=False,
    ):
        """Maximize the function using the optimizer."""
        self.clearMemoization()
        optimizer_instance = OptimizerFactory().get_optimizer(optimizer=optimizer)
        return optimizer_instance.optimize(
            self,
            budget=budget,
            stopIfZeroGain=stopIfZeroGain,
            stopIfNegativeGain=stopIfNegativeGain,
            epsilon=epsilon,
            verbose=verbose,
            show_progress=show_progress,
        )

    def marginalGain(self, X, element):
        """Compute the marginal gain of adding an element to the set."""
        element = self._normalize_element(element)
        if self._contains(X, element) or not self._valid_element(element):
            return 0.0

        curr_val = self.evaluate(X)
        X_new = set(X) if X is not None and not isinstance(X, torch.Tensor) else set(
            self._indices_from_set(X).detach().cpu().tolist()
        )
        X_new.add(element)
        new_val = self.evaluate(X_new)
        return new_val - curr_val

    def evaluate(self, evaluate_set):
        """Evaluate the function on the given set."""
        if self._is_empty_selection(evaluate_set):
            return 0.0
        selected_max = self._selected_max(evaluate_set)
        return self._score_from_memo(selected_max)

    def marginalGainWithMemoization(self, X, element):
        """Compute the marginal gain of adding an element with memoization."""
        if not self.memoization_initialized:
            return self.marginalGain(X, element)

        element = self._normalize_element(element)
        if self._contains(X, element) or not self._valid_element(element):
            return 0.0

        memo = self.similarity_with_nearest_in_effective_x
        candidate_sim = self.data_sijs[:, element]
        new_best = torch.maximum(memo, candidate_sim)
        return (self._score_from_memo(new_best) - self._score_from_memo(memo)).item()

    def batchedGain(self, X):
        X = X.to(device=self.device, dtype=torch.bool).flatten()
        if X.numel() != self.n:
            raise Exception(f"ERROR: Boolean selected mask should have length {self.n}")
        remaining = (~X).nonzero(as_tuple=False).flatten()
        if remaining.numel() == 0:
            return (
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                torch.tensor(-1, dtype=torch.long, device=self.device),
            )

        memo = self.similarity_with_nearest_in_effective_x
        candidate_sims = self.data_sijs[:, remaining]
        new_best = torch.maximum(memo.unsqueeze(1), candidate_sims)

        old_score = torch.clamp(
            torch.minimum(memo, self.query_cap) - self.private_penalty, min=0
        )
        new_score = torch.clamp(
            torch.minimum(new_best, self.query_cap.unsqueeze(1))
            - self.private_penalty.unsqueeze(1),
            min=0,
        )
        gains = (new_score - old_score.unsqueeze(1)).sum(dim=0)
        best_gain = gains.max()
        tied_rel_indices = torch.where(gains == best_gain)[0]
        rel_idx = tied_rel_indices[-1]
        best_idx = remaining[rel_idx]
        return best_gain, best_idx

    def updateBatchMemo(self, element):
        element = self._normalize_element(element)
        if not self._valid_element(element):
            return
        candidate = self.data_sijs[:, element]
        self.similarity_with_nearest_in_effective_x = torch.maximum(
            self.similarity_with_nearest_in_effective_x, candidate
        )

    def evaluateWithMemoization(self, evaluate_set):
        """Evaluate the function on the given set with memoization."""
        if self._is_empty_selection(evaluate_set):
            return 0.0
        if not self.memoization_initialized:
            return self.evaluate(evaluate_set)
        return self._score_from_memo(self.similarity_with_nearest_in_effective_x).item()

    def updateMemoization(self, X, element):
        """Update memoization after adding an element."""
        if not self.memoization_initialized:
            return

        element = self._normalize_element(element)
        if self._contains(X, element) or not self._valid_element(element):
            return

        candidate = self.data_sijs[:, element]
        torch.maximum(
            self.similarity_with_nearest_in_effective_x,
            candidate,
            out=self.similarity_with_nearest_in_effective_x,
        )

    def clearMemoization(self):
        """Clear the selected-set memoization."""
        if self.memoization_initialized:
            self.similarity_with_nearest_in_effective_x.zero_()

    def setMemoization(self, X):
        """Set memoization for the given selected set."""
        if not self.memoization_initialized:
            return

        self.clearMemoization()
        indices = self._indices_from_set(X)
        if indices.numel() == 0:
            return
        selected_max = torch.clamp(self.data_sijs[:, indices], min=0).max(dim=1).values
        self.similarity_with_nearest_in_effective_x.copy_(selected_max)

    def getEffectiveGroundSet(self):
        """Get the effective ground set."""
        return self.effective_ground_set
