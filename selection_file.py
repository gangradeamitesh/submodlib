import torch
import torch.nn.functional as F
from vectorized_greedy import maximize_function

# from submodlib import FacilityLocationVariantMutualInformationFunction
# from submodlib import FacilityLocationVariantMutualInformation
from submodlib import FacilityLocationMutualInformation
from submodlib import FacilityLocationMutualInformationFunction

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
    

    obj = FacilityLocationMutualInformation(n=data.shape[0], num_queries=query.shape[0], magnificationEta=1.0 ,data=data , query_data=query, metric="cosine")

    greedy_list = obj.maximize(optimizer="NaiveGreedy" , budget=BUDGET , stopIfZeroGain=False , stopIfNegativeGain =False, epsilon=False , verbose=False , show_progress=False , costs=None , costSensitiveGreedy=False)
    print("---------------")
    print(greedy_list)


    print("C++ --------------------")
    obj = FacilityLocationMutualInformationFunction(n=data.shape[0], num_queries=query.shape[0], data=data, 
                                                    queryData=query, metric="cosine", 
                                                    magnificationEta=1.0)
    greedyList = obj.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
    print(greedyList)