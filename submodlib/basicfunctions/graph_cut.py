from pickle import NONE
from base_function import BaseFunction
from optimizers.optimizer_factory import OptimizerFactory
class GraphCut(BaseFunction):

    def __init__(self, n, mode="dense", seperate_rep=None, n_rep=None, sijs=None, 
                 data=None, data_rep=None, num_clusters=None, cluster_labels=None, 
                 metric="cosine", num_neighbors=None, create_dense_cpp_kernel_in_python=True, 
                 pybind_mode=None, partial=False, ground_set=None, separate_master=False ) -> None:    
        super().__init__(n=n, mode=mode, sijs=sijs, data=data,cluster_label=cluster_labels , num_clusters=num_clusters, metric=metric)
        self.n_rep = n_rep
        
        self.data_rep = data_rep
        self.num_neighbors = num_neighbors
        self.separate_rep = seperate_rep
        self.effective_ground = None
        self.create_dense_kernel = create_dense_cpp_kernel_in_python
        self.optimizer = None
        
        # New parameters for proper ground set handling
        self.partial = partial
        self.ground_set = ground_set
        self.separate_master = separate_master
        
        # Memoization variables (similar to C++ version)
        self.similarity_with_nearest_in_effective_x = None
        self.memoization_initialized = False
        
        # Effective ground set - this is what getEffectiveGroundSet() should return
        self.effective_ground_set = None
        self.master_set = None
        self.n_master = None
        self._initialize_ground_sets()
    
    def _initialize_ground_sets(self):
        """Initialize effective ground set and master set like C++ version"""
        if self.partial and self.ground_set is not None:
            # Use provided ground set (partial mode)
            self.effective_ground_set = set(self.ground_set)
        else:
            # Create ground set with items 0 to n-1 (like C++ lines 28-32)
            self.effective_ground_set = set(range(self.n))
    
    def maximize(self, optimizer, budget, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=None, 
                 verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
        """Maximize the function using the optimizer"""
        optimizer_instance = OptimizerFactory.get_optimizer(optimizer)
        return optimizer_instance.optimize(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)
        pass
    
    def marginalGain(self , X , element):
        """Compute the marginal gain of adding an element to the set"""
        pass
    
    def evaluate(self , evaluate_set):
        """Evalaute the function on the given set"""
        pass 
    
    def marginalGainWithMemoization(self , X , element):
        """Compute the marginal gain of adding an element to the set with memoization"""
        pass
    
    def evaluateWithMemoization(self , evaluate_set):
        """Evaluate the function on the given set with memoization"""
        pass
    
    def updateMemoization(self , X):
        """Update the memoization for the given set"""
        pass
    
    def clearMemoization(self):
        """Clear the memoization"""
        pass
    
    def setMemoization(self , X):
        """Set the memoization for the given set"""
        pass
    
    def getEffectiveGroundSet(self):
        """Get the effective ground set"""
        return self.effective_ground_set.copy()

if __name__ == "__main__":
    print("Testing Facility Location Implementation")
    from sklearn.datasets import make_blobs
    import random
    num_clusters = 10
    cluster_std_dev = 4
    points, cluster_ids, centers = make_blobs(n_samples=500, centers=num_clusters, 
                                            n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), 
                                            return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]
    import numpy as np
    dataArray = np.array(data)
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, 6)
    subset1xs = [xs[x] for x in subset1]
    subset1ys = [ys[x] for x in subset1]
    set1 = set(subset1[:-1])
    subset2 = []
    for i in range(6):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    subset2xs = [xs[x] for x in subset2]
    subset2ys = [ys[x] for x in subset2]
    set2 = set(subset2[:-1])
    obj1 = GraphCut(n=500, mode="dense", data=dataArray, metric="euclidean")
    obj1.maximize(budget=1, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(f"Subset 1's FL value = {obj1.evaluate(set1)}")
    print(f"Subset 2's FL value = {obj1.evaluate(set2)}")
    print(f"Gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj1.marginalGain(set1, subset1[-1])}")
    print(f"Gain of adding another point ({subset2[-1]}) of different cluster to {set1} = {obj1.marginalGain(set1, subset2[-1])}")
    obj1.setMemoization(set1)
    print(f"Subset 1's Fast FL value = {obj1.evaluateWithMemoization(set1)}")
    print(f"Fast gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj1.marginalGainWithMemoization(set1, subset1[-1])}")
    #start = time.process_time()
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    #print(f"Time taken by maximization = {time.process_time() - start}")
    print(f"Greedy vector: {greedyList}")
    greedyXs = [xs[x[0]] for x in greedyList]
    greedyYs = [ys[x[0]] for x in greedyList]