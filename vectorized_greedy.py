import torch
import tqdm

def maximize_function(function, budget):
    n = function.n
    device = function.device
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    picked = []

    function.clearMemoization()
    current_best = function.similarity_with_nearest_in_effective_x  

    iterator = range(budget)
    progress = None

    from tqdm.auto import tqdm

    for step in tqdm(iterator, total=budget, desc="Vector Greedy", leave=True):
        #for j in num_chunk:
        gains = torch.maximum(current_best, function.query_sijs) - current_best 
        gains = gains.sum(dim=1) + function.queryDiversityEta * function.query_cap
        gains = gains.masked_fill(selected, float("-inf"))

        best_gain, best_idx = gains.max(dim=0)
        best_idx = int(best_idx.item())

        picked.append((best_idx, best_gain))
        selected[best_idx] = True

        candidate = function.query_sijs[best_idx]
        current_best = torch.maximum(current_best, candidate)
        function.similarity_with_nearest_in_effective_x = current_best
    
    
    return picked
