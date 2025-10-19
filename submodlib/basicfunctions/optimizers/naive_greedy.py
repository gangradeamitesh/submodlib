from .base_optimizer import BaseOptimizer

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
        selected = set()
        selected_pairs = set()
        ground_set = function.getEffectiveGroundSet()
        
        # Initialize memoization for the empty set
        function.setMemoization(selected)
        
        for iteration in range(budget):
            if verbose:
                print(f"Iteration {iteration + 1}/{budget}")
            
            best_element = None
            best_gain = -float('inf')
            
            # Find element with maximum marginal gain
            for element in ground_set:
                if element in selected:
                    continue
                
                # Use memoized marginal gain if available
                if hasattr(function, 'marginalGainWithMemoization'):
                    gain = function.marginalGainWithMemoization(selected, element)
                else:
                    gain = function.marginalGain(selected, element)
                
                if gain > best_gain:
                    best_gain = gain
                    best_element = element
                    output_pair = (element, gain)
            
            # Check stopping conditions
            if best_element is None:
                if verbose:
                    print("No more elements to select")
                break
            
            if stopIfZeroGain and best_gain <= 0:
                if verbose:
                    print(f"Stopping: marginal gain is {best_gain} (zero or negative)")
                break
            
            if stopIfNegativeGain and best_gain < 0:
                if verbose:
                    print(f"Stopping: marginal gain is {best_gain} (negative)")
                break
            
            # Add best element to selected set
            selected.add(best_element)
            selected_pairs.add(output_pair)
            
            # Update memoization
            if hasattr(function, 'updateMemoization'):
                function.updateMemoization(selected, best_element)
            
            if verbose:
                current_value = function.evaluateWithMemoization(selected) if hasattr(function, 'evaluateWithMemoization') else function.evaluate(selected)
                print(f"Selected element {best_element} with gain {best_gain:.4f}, current value: {current_value:.4f}")
        
        if verbose:
            final_value = function.evaluateWithMemoization(selected) if hasattr(function, 'evaluateWithMemoization') else function.evaluate(selected)
            print(f"Optimization complete. Selected {len(selected)} elements with final value: {final_value:.4f}")
        
        return list(selected_pairs)
