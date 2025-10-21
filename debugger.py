from submodlib import FacilityLocation
import torch

if __name__ == "__main__":
    l = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.float32)
    obj = FacilityLocation(n = len(l), mode = "dense" , data = l , metric="euclidean")
    result = obj.maximize(optimizer="NaiveGreedy" ,budget=3 , stopIfZeroGain=False , stopIfNegativeGain=False , verbose=False)
    print(f"Optimization result: {result}")