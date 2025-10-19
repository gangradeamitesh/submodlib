from pickle import NONE
from base_function import BaseFunction
class GraphCut(BaseFunction):

    def __init__(self , n , mode="dense" ,seperate_rep = None , n_rep = None , sijs = None , 
    data = None ,data_rep=None, num_clusters=None, cluster_label=None,
    metric="cosine", num_neighbors=None, create_dense_cpp_kernel_in_python=True, 
    partial=False, ground_set=None, separate_master=False ) -> None:
    

        super().__init__()