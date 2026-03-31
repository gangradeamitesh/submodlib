
import torch
import torch.nn.functional as F

class DenseSimilarity:

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
    
    @staticmethod
    def rbf_similarity(ground_set, candidate):
        # Compute pairwise squared Euclidean distances
        sq_dists = torch.cdist(ground_set, candidate, p=2).pow(2)
        # Compute the RBF (Radial Basis Function) kernel
        gamma = 1.0 / ground_set.size(-1)  # Gamma is the inverse of the number of features
        similarity = torch.exp(-gamma * sq_dists)
        return similarity
