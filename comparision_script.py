import torch
from submodlib import FacilityLocationVariantMutualInformationFunction
import torch.nn.functional as F
from vectorized_greedy import maximize_function

from submodlib import FacilityLocationVariantMutualInformation

data = torch.randn(1000, 1024 ,dtype=torch.float16)
query = torch.randn(800, 1024 , dtype=torch.float16)