
import torch
import torch.nn.functional as F

class DenseSimilarity:

    def __init__(self) -> None:
        pass

    @staticmethod
    def euclidean_distance(data):
        """Compute the pairwise Euclidean distance between data points."""

        return torch.cdist(data, data, p=2)
    
    @staticmethod
    def consine_similarity(data):
        """Compute the pairwise cosine similarity between data points."""

        data = F.normalize(data, p=2, dim=1)
        return torch.mm(data, data.t())
    