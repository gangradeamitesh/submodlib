# from submodlib import FacilityLocation
# import torch

# if __name__ == "__main__":
#     print("Testing Facility Location Implementation")
#     from sklearn.datasets import make_blobs
#     num_clusters = 10
#     cluster_std_dev = 4
#     points, cluster_ids, centers = make_blobs(n_samples=500, centers=num_clusters, 
#                                             n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), 
#                                             return_centers=True, random_state=4)
#     data = list(map(tuple, points))
#     xs = [x[0] for x in data]
#     ys = [x[1] for x in data]
#     # get 6 data points belonging to cluster#1
#     import random
#     random.seed(1)
#     cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
#     subset1 = random.sample(cluster1Indices, 6)
#     subset1xs = [xs[x] for x in subset1]
#     subset1ys = [ys[x] for x in subset1]
#     set1 = set(subset1[:-1])
#     # get 6 data points belonging to different clusters
#     subset2 = []
#     for i in range(6):
#         #find the index of first point that belongs to cluster i
#         diverse_index = cluster_ids.tolist().index(i)
#         subset2.append(diverse_index)
#     subset2xs = [xs[x] for x in subset2]
#     subset2ys = [ys[x] for x in subset2]
#     set2 = set(subset2[:-1])
#     import numpy as np
#     dataArray = np.array(data)
#     #start = time.process_time()
#     obj1 = FacilityLocation(n=500, mode="dense", data=dataArray, metric="euclidean")
#     print(obj1.sijs)

#     #print(f"Time taken by instantiation = {time.process_time() - start}")
#     print(f"Subset 1's FL value = {obj1.evaluate(set1)}")
#     print(f"Subset 2's FL value = {obj1.evaluate(set2)}")
#     print(f"Gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj1.marginalGain(set1, subset1[-1])}")
#     print(f"Gain of adding another point ({subset2[-1]}) of different cluster to {set1} = {obj1.marginalGain(set1, subset2[-1])}")
#     obj1.setMemoization(set1)
#     print(f"Subset 1's Fast FL value = {obj1.evaluateWithMemoization(set1)}")
#     print(f"Fast gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj1.marginalGainWithMemoization(set1, subset1[-1])}")
#     #start = time.process_time()
#     greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
#     #print(f"Time taken by maximization = {time.process_time() - start}")
#     print(f"Greedy vector: {greedyList}")
#     greedyXs = [xs[x[0]] for x in greedyList]
#     greedyYs = [ys[x[0]] for x in greedyList]
from submodlib import FacilityLocation
import torch
if __name__ == "__main__":
    # rows = masters, cols = ground items
    sijs = torch.Tensor([
  [1.0, 0.2, 0.4],
  [0.2, 1.0, 0.3],
  [0.4, 0.3, 1.0],
])
    obj = FacilityLocation(n=3, mode="dense", sijs=sijs)
    print("Testing Facility Location Implementation")
    greedyList = obj.maximize(budget=2, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=True)
    print(f"Greedy vector: {greedyList}")