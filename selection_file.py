import torch
import torch.nn.functional as F
from vectorized_greedy import maximize_function

# from submodlib import FacilityLocationVariantMutualInformationFunction
# from submodlib import FacilityLocationVariantMutualInformation
# from submodlib import FacilityLocationMutualInformation
# from submodlib import FacilityLocationMutualInformationFunction
from submodlib import FacilityLocationFunction
from submodlib import FacilityLocation
from submodlib import GraphCutFunction
from submodlib import GraphCut

#from submodlib import faci
from submodlib import GraphCutConditionalGain
from submodlib import GraphCutConditionalGainFunction
# data_dict = torch.load("p3_data_dict.pt")
# device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# query_dict = torch.load("query.pt")
# ground_all_features = []
# ground_all_image_ids = []
# for img_id , feat in data_dict.items():
#     ground_all_features.append(feat)
#     ground_all_image_ids.extend([img_id]*feat.shape[0])
# valid_data = [f for f in ground_all_features if f.numel() > 0]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = torch.cat(valid_data , dim=0).to(device)

# query = torch.cat([
#     v for v in query_dict.values() if v.numel()>0
# ],
# dim=0).to(device)

def _mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


device = torch.device("mps" if _mps_available() else "cpu")
data = torch.randn(1000, 1024 ,dtype=torch.float16)
query = torch.randn(800, 1024 , dtype=torch.float16 , device=device)
print("Data Matrix device")
print(data.device)
print("Query Matrix device")
print(query.device)

num_chunk = 10
CHUNK_SIZE = data.shape[0]/num_chunk
BUDGET = 10


def do_selection(budget , data, query_data,device):
    data = data.to(device)
    obj2 = FacilityLocationVariantMutualInformation(n=data.shape[0], num_queries=query_data.shape[0], queryDiversityEta=1.0 ,data=data , query_data=query, metric="cosine")

    #free , total = torch.cuda.mem_get_info()
    # print("function device")
    # print(obj2.device)
    # print("Query sij device")
    # print(obj2.query_sijs.device)
    # print("Query sij shape")
    # print(obj2.query_sijs.shape)

    result = maximize_function(obj2, budget=budget)
    #print(result)
    selected_idx = [r[0] for r in result]
    #print(selected_idx)
    return result, data[selected_idx].cpu()


def cosine_similarity(ground_set, candidate):
    ground_norm = F.normalize(ground_set, p=2, dim=1).to(device)
    candidate_norm = F.normalize(candidate, p=2, dim=1)
    return torch.matmul(ground_norm , candidate_norm.T)


def main():
    d = {}
    i = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_data = torch.empty(0,1024)
    for chunk in torch.split(data , int(CHUNK_SIZE)):
        res , feat = do_selection(budget=BUDGET,data=chunk,query_data=query,device=device)
        selected_data = torch.cat([selected_data ,feat])
        print(res)
        for r in res:
            d[i] = r[0]
            i += 1

    print(selected_data.shape)
    print("Device of selected data")
    
    result , select_feat = do_selection(budget=BUDGET , data=selected_data.to(torch.float16) , query_data=query,device=device)
    # print(len(result))
    # print(select_feat)
    print(result)
    return select_feat
    

if __name__ == "__main__":
    

    import numpy as np
    import matplotlib.pyplot as plt
    groundData =np.array( [(3,12.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
    (4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
    (4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
    (4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
    (4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
    (4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (7.5,10), (12.5,10), (10,12.5), 
    (10,7.5), (8,12.5), (8,7.5), (14,12.5), (14,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )
    print("Number of elements in ground set = ", len(groundData))
    groundxs = [x[0] for x in groundData]
    groundys = [x[1] for x in groundData]

    mutlipleQueryData = np.array([(4.5,13.5), (15.5,6.5)])
    multiplequeryxs = [x[0] for x in mutlipleQueryData]
    multiplequeryys = [x[1] for x in mutlipleQueryData]

    mutlipleQueryData2 = np.array([(4.5,13.5), (15.5,11)])
    multiplequeryxs2 = [x[0] for x in mutlipleQueryData2]
    multiplequeryys2 = [x[1] for x in mutlipleQueryData2]

    singleQueryData = np.array([(4.5,13.5)])
    singlequeryxs = [x[0] for x in singleQueryData]
    singlequeryys = [x[1] for x in singleQueryData]

    from submodlib import FacilityLocationConditionalGainFunction
    from submodlib import FacilityLocationConditionalGain

    nus = [0, 0.3, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 10, 50, 100]
    row = 0
    index = 1
    plt.figure(figsize = (16, 16))
    for nu in nus:
        print("C++ ------------------------------")
        obj = GraphCutConditionalGainFunction(n=46, 
                                            num_privates=1, 
                                            data=groundData,
                                            privateData=singleQueryData,
                                            metric="cosine",
                                            lambdaVal=0.5, 
                                            privacyHardness=nu,)
        greedyList = obj.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                                stopIfNegativeGain=False, verbose=False)
        greedyXs = [groundxs[x[0]] for x in greedyList]
        greedyYs = [groundys[x[0]] for x in greedyList]
        
        print("Pytorch --------------------------------")

        obj2 = GraphCutConditionalGain(n=46, 
                                            num_privates=1, 
                                            data=groundData,
                                            privateData=singleQueryData,
                                            metric="cosine", 
                                            privacyHardness=nu,lambdaVal=0.5)
        greedyList2 = obj2.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                                stopIfNegativeGain=False, verbose=False)
        
        idxs_cpp = torch.tensor([p[0] for p in greedyList], dtype=torch.long)
        idxs_py  = torch.tensor([p[0] for p in greedyList2], dtype=torch.long)

# gains may be floats or 0-d tensors in different dtypes
        gains_cpp = torch.tensor([float(p[1]) for p in greedyList], dtype=torch.float16)
        gains_py  = torch.tensor([float(p[1]) for p in greedyList2], dtype=torch.float16)

        same_indices = torch.equal(idxs_cpp, idxs_py)
        same_gains = torch.allclose(gains_cpp, gains_py, rtol=1e-5, atol=1e-7)

        print("indices match:", same_indices)
        print("gains match:", same_gains)
        # if not same_indices or not same_gains:
        #     for i, (c, p) in enumerate(zip(greedyList, greedyList2)):
        #         print(f"{i}: C++ {c}, PyTorch {p}")
        for i , j in zip(greedyList , greedyList2):
            #assert greedyList[i][0] == greedyList2[i][0] , "Selected indices do not match"
            #assert abs(greedyList[i][1] - greedyList2[i][1]) < 1e-5 , "Gains do not match"
            #print(f"Index {i} matches: {greedyList[i]} == {greedyList2[i]}")
            print("Gain difference:", (greedyList[i][1],greedyList2[i][1]))
    