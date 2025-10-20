
import torch
import torch.nn.functional as F

class DenseSimilarity:

    def __init__(self) -> None:
        pass

    @staticmethod
    def euclidean_distance(data, candidate_data , sigma=1.0):
        """Compute the pairwise Euclidean distance between data points."""
        dist = torch.cdist(data, candidate_data, p=2)**2
        sim = torch.exp(-dist / (2 * sigma ** 2))
        return sim
    
    @staticmethod
    def cosine_similarity(data,candidate_data):
        """Compute the pairwise cosine similarity between data points."""
        data = F.normalize(data, p=2, dim=1)
        return torch.matmul(data, candidate_data.T)