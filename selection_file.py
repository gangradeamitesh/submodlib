import torch
import torch.nn.functional as F
from vectorized_greedy import maximize_function
from submodlib import FacilityLocationFunction
from submodlib import FacilityLocation
from submodlib import GraphCutFunction
from submodlib import GraphCut

#from submodlib import faci
from submodlib import GraphCutConditionalGain
from submodlib import GraphCutConditionalGainFunction
from submodlib import FacilityLocationVariantMutualInformation
from submodlib import FacilityLocationVariantMutualInformationFunction

    

if __name__ == "__main__":
    
    import torch
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    import numpy as np
    import matplotlib.pyplot as plt
    # groundData =np.array( [(3,12.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
    # (4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
    # (4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
    # (4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
    # (4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
    # (4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (7.5,10), (12.5,10), (10,12.5), 
    # (10,7.5), (8,12.5), (8,7.5), (14,12.5), (14,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )
    # print("Number of elements in ground set = ", len(groundData))
    # groundxs = [x[0] for x in groundData]
    # groundys = [x[1] for x in groundData]

    # mutlipleQueryData = np.array([(4.5,13.5), (15.5,6.5)])
    # multiplequeryxs = [x[0] for x in mutlipleQueryData]
    # multiplequeryys = [x[1] for x in mutlipleQueryData]

    # mutlipleQueryData2 = np.array([(4.5,13.5), (15.5,11)])
    # multiplequeryxs2 = [x[0] for x in mutlipleQueryData2]
    # multiplequeryys2 = [x[1] for x in mutlipleQueryData2]

    # singleQueryData = np.array([(4.5,13.5)])
    # singlequeryxs = [x[0] for x in singleQueryData]
    # singlequeryys = [x[1] for x in singleQueryData]

    # from submodlib import FacilityLocationConditionalGainFunction
    # from submodlib import FacilityLocationConditionalGain

    groundData = torch.randn(100000,1024)
    singleQueryData = torch.randn(800,1024)

    #nus = [0, 0.3, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 10, 50, 100]
    nus = [0.3]
    row = 0
    index = 1
    plt.figure(figsize = (16, 16))
    for nu in nus:
        print("C++ ------------------------------")
        # obj = FacilityLocationVariantMutualInformationFunction(n=groundData.shape[0], 
        #                                     num_queries=singleQueryData.shape[0],
        #                                     data=groundData,
        #                                     queryData=singleQueryData,
        #                                     metric="cosine",
                                            
        #                                     queryDiversityEta=nu,)
        # greedyList = obj.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
        #                         stopIfNegativeGain=False, verbose=False)
        # print(greedyList)
        
        print("Pytorch --------------------------------")

        obj2 = FacilityLocationVariantMutualInformation(n=groundData.shape[0], 
                                            num_queries=singleQueryData.shape[0], 
                                            data=groundData,
                                            query_data=singleQueryData,
                                            metric="cosine", 
                                            queryDiversityEta=nu)
        greedyList2 = obj2.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                                stopIfNegativeGain=False, verbose=False)
        print(greedyList2)
        idxs_cpp = torch.tensor([p[0] for p in greedyList], dtype=torch.long)
        idxs_py  = torch.tensor([p[0] for p in greedyList2], dtype=torch.long)

        gains_cpp = torch.tensor([float(p[1]) for p in greedyList], dtype=torch.float16)
        gains_py  = torch.tensor([float(p[1]) for p in greedyList2], dtype=torch.float16)

        same_indices = torch.equal(idxs_cpp, idxs_py)
        same_gains = torch.allclose(gains_cpp, gains_py, rtol=1e-5, atol=1e-7)

        print("indices match:", same_indices)
        print("gains match:", same_gains)
        # if not same_indices or not same_gains:
        #     for i, (c, p) in enumerate(zip(greedyList, greedyList2)):
        #         print(f"{i}: C++ {c}, PyTorch {p}")
        # for i , j in zip(greedyList , greedyList2):
        #     #assert greedyList[i][0] == greedyList2[i][0] , "Selected indices do not match"
        #     #assert abs(greedyList[i][1] - greedyList2[i][1]) < 1e-5 , "Gains do not match"
        #     #print(f"Index {i} matches: {greedyList[i]} == {greedyList2[i]}")
        #     print("Gain difference:", (greedyList[i][1],greedyList2[i][1]))
    
