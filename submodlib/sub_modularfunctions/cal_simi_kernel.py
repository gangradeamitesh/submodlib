
import torch
import torch.nn.functional as F

class DenseSimilarity:

    def __init__(self) -> None:
        pass

    @staticmethod
    def euclidean_distance(data, candidate_data , sigma=1.0):
        dist = torch.cdist(data, candidate_data, p=2)
        feature_dim = data.size(-1)
        gamma = 1.0 / feature_dim
        sim = torch.exp(-dist * gamma)
        return sim
    
    @staticmethod
    def cosine_similarity(ground_set, candidate):
        ground_norm = F.normalize(ground_set, p=2, dim=1)
        candidate_norm = F.normalize(candidate, p=2, dim=1)
        return torch.matmul(ground_norm , candidate_norm.T)
