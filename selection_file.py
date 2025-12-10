import torch
import torch.nn.functional as F
from vectorized_greedy import maximize_function

# from submodlib import FacilityLocationVariantMutualInformationFunction
# from submodlib import FacilityLocationVariantMutualInformation
# from submodlib import FacilityLocationMutualInformation
# from submodlib import FacilityLocationMutualInformationFunction
# from submodlib import FacilityLocationFunction
# from submodlib import FacilityLocation
# from submodlib import GraphCutFunction
# from submodlib import GraphCut

# #from submodlib import faci
# from submodlib import GraphCutConditionalGain
# from submodlib import GraphCutConditionalGainFunction
# from submodlib import LogDeterminantMutualInformationFunction
# from submodlib import LogDeterminantMutualInformation
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import FacilityLocationVariantMutualInformation

if __name__ == "__main__":

    import torch

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
    
    print("Running selection")
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

    #nus = [0, 0.3, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 10, 50, 100]
    nus = [0.3]
    row = 0
    index = 1
    plt.figure(figsize = (16, 16))
    for nu in nus:
        print("C++ ------------------------------")
        obj = FacilityLocationVariantMutualInformationFunction(n=46, 
                                            num_queries=1, 
                                            data=groundData,
                                            queryData=singleQueryData,
                                            metric="cosine",
                                            # lambdaVal = 1
                                            )
        greedyList = obj.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                                stopIfNegativeGain=False, verbose=False)
        print(greedyList)
        greedyXs = [groundxs[x[0]] for x in greedyList]
        greedyYs = [groundys[x[0]] for x in greedyList]
        
        print("Pytorch --------------------------------")

        obj2 = FacilityLocationVariantMutualInformation(n=46, 
                                            num_queries=1, 
                                            data=groundData,
                                            queryData=singleQueryData,
                                            metric="cosine", 
                                            device = "mps"
                                            )
        print(obj2.device)
        greedyList2 = obj2.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                                stopIfNegativeGain=False, verbose=False , epsilon=1e-8 , show_progress=True,costs = None,costSensitiveGreedy=False)
        print(greedyList2)
         
    