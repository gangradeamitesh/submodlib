from .base_optimizer import BaseOptimizer
import torch

class NaiveGreedy(BaseOptimizer):
    """
    Naive Greedy optimizer implementation.
    
    At each step, selects the element with maximum marginal gain.
    """

    def __init__(self):
        pass

    def optimize(self, function, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True):
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
            zero = torch.tensor(0.0, device=function.device)
            if stopIfNegativeGain and torch.lt(best_gain, zero):
                if verbose:
                    print("Stopping early due to negative gain.")
                break
            if stopIfZeroGain and torch.eq(best_gain, zero):
                if verbose:
                    print("Stopping early due to zero gain.")
                break
            #best_idx = int(best_idx.item())
            selected[best_idx] = True
            selected_pairs.append((best_idx , best_gain))
            function.updateBatchMemo(best_idx)
            
        return [(int(idx.item()), float(gain.item())) for idx, gain in selected_pairs]
