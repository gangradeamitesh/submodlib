from userValidator import *
import torch as t
from cal_simi_kernel import DenseSimilarity
from base_function import BaseFunction
from optimizers.optimizer_factory import OptimizerFactory

class FacilityLocation(BaseFunction):
    """"Similarity kernel , data should be a torch kernel
    and if provided as numpy arrray. 
    1.Add functioality to convert the numpy array to torch tensor"""

    def __init__(self , n , mode , separate_rep=None , n_rep = None , sijs=None , data = None , data_rep = None , num_clusters=None , cluster_labels=None , metric="cosine" , num_neighbors = None , create_dense_cpp_kernel_in_python=None , pybind_mode=None) -> None:
        self.n = n
        self.n_rep = n_rep
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.data_rep=data_rep
        self.num_neighbors = num_neighbors
        """Currently not supporting spearate representation"""
        self.separate_rep=separate_rep
        self.clusters=None
        self.cluster_sijs=None
        self.cluster_map=None
        self.cluster_labels=cluster_labels
        self.num_clusters=num_clusters
        self.cpp_obj = None
        self.cpp_sijs = None
        self.cpp_ground_sub = None
        self.cpp_content = None
        self.effective_ground = None
        self.create_dense_kernel = create_dense_cpp_kernel_in_python
        self.optimizer = None

        """Validating the input"""
        validate_n(self.n)
        validate_mode(self.mode)
        validate_sep_rep(self.separate_rep ,self.mode , self.n_rep)

        """Currently not supporting custered mode"""
        #Similarity kernel provided
        if type(self.sijs) != type(None):
            validate_sijs(type(self.sijs) , self.mode , self.num_neighbors , self.separate_rep)
            """"Dimensionality check"""
            if self.separate_rep==True:
                if self.data.shape[1] != self.data_rep.shape[1]:
                    raise Exception("ERROR: Data and Representation have different dimensions")
            else:
                if np.shape(self.data)[1] != np.shape(self.data)[1]:
                    raise Exception("ERROR: Data and Representation have different dimensions")
            if type(self.data) != type(None) or type(self.data_rep) != type(None):
                print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else:#similarity kernel not provided
            if self.data == None:
                raise Exception("ERROR: Data matrix not provided")
            if self.create_dense_kernel==True and self.mode=="dense" and self.metric=="euclidean":
                self.sijs = DenseSimilarity.euclidean_distance(self.data)
            elif self.create_dense_kernel==True and self.mode=="dense" and self.metric=="cosine":
                self.sijs = DenseSimilarity.cosine_similarity(self.data)
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
        
        def maximize(self , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
            """Maximize the function using the optimizer"""
            self.optimizer = OptimizerFactory.getOptimizer(optimizer)
            output = self.optimizer.optimiz(self , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy)
            return output
        def marginalGain(self , X , element):
            print("Marginal Gain")
            pass
        def evaluate(self , evaluate_set):
            pass
        def marginalGainWithMemoization(self , X , element):
            pass
        def evalauteWithMemoization(self , evaluate_set):
            pass
        def updateMemoization(self , X):
            pass
        def clearMemoization(self):
            pass
        def setMemoization(self , X):
            pass
        def getEffectiveGroundSet(self):
            pass


if __name__ == "__main__":
    print("Running the main function")
    from sklearn.datasets import make_blobs
    num_clusters = 10
    cluster_std_dev = 4
    points, cluster_ids, centers = make_blobs(n_samples=500, centers=num_clusters, 
                                          n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), 
                                          return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]
    dataArray = np.array(data)
    fl_object = FacilityLocation(n=500, mode="dense", data=dataArray, metric="euclidean")
    print(fl_object.maximize(optimizer="NaiveGreedy", budget=10, stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False))