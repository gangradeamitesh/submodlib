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
        
        # Initialize
        selected = torch.zeros(function.n , dtype=torch.bool, device=function.device)
        selected_pairs = []
        ground_set = function.getEffectiveGroundSet()
        if isinstance(ground_set, torch.Tensor):
            ground_set = ground_set.tolist()
        
        # Initialize memoization for the empty set
        function.setMemoization(selected)

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
            
            best_gain = -float('inf')
            
            # for element in ground_set:
            #     if element in selected:
            #         continue
                
            #     """TODO: This is commented memoization code, to be revisited later"""
            #     # Use memoized marginal gain if available
            #     if hasattr(function, 'marginalGainWithMemoization'):
            #         gain = function.marginalGainWithMemoization(selected, element)
            #     else:
            #         gain = function.marginalGain(selected, element)
                
            #     if gain >= best_gain:
            #         best_gain = gain
            #         best_element = element
            #         """TODO: This is a temporary fix to return gain along with element"""
            #         output_pair = (element, gain) 
            best_gain , best_idx = function.batchedGain(selected)
            best_idx = int(best_idx.item())
            selected[best_idx] = True
            selected_pairs.append((best_idx , best_gain))
            function.updateBatchMemo(best_idx)
            
            # if best_element is None:
            #     if verbose:
            #         print("No more elements to select")
            #     break
            
            # if stopIfZeroGain and best_gain <= 0:
            #     if verbose:
            #         print(f"Stopping: marginal gain is {best_gain} (zero or negative)")
            #     break
            
            # if stopIfNegativeGain and best_gain < 0:
            #     if verbose:
            #         print(f"Stopping: marginal gain is {best_gain} (negative)")
            #     break
            

            """TODO: This is commented memoization code, to be revisited later"""
            # if hasattr(function, 'updateMemoization'):
            #     function.updateMemoization(selected , best_element)
            # selected.add(best_element)
            # selected_pairs.add(output_pair)
            # if progress is not None:
            #     progress.set_postfix({"gain": float(best_gain)})
            # if verbose:
            #     current_value = function.evaluateWithMemoization(selected) if hasattr(function, 'evaluateWithMemoization') else function.evaluate(selected)
            #     print(f"Selected element {best_element} with gain {best_gain:.4f}, current value: {current_value:.4f}")
        
        # if progress is not None:
        #     progress.close()

        # if verbose:
        #     final_value = function.evaluateWithMemoization(selected) if hasattr(function, 'evaluateWithMemoization') else function.evaluate(selected)
        #     print(f"Optimization complete. Selected {len(selected)} elements with final value: {final_value:.4f}")
        
        return selected_pairs
