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
from submodlib import LogDeterminantMutualInformationFunction
from submodlib import LogDeterminantMutualInformation
from submodlib import FacilityLocationConditionalGainFunction
from submodlib import FacilityLocationConditionalGain
import time

def compare_flmi(ground_data, query_data, budget):
    # C++ Implementation
    flmi_cpp = FacilityLocationVariantMutualInformationFunction(n=ground_data.shape[0],
                                                               num_queries=query_data.shape[0],
                                                               data=ground_data,
                                                               queryData=query_data,
                                                               metric="cosine")
    start_cpp = time.perf_counter()
    greedy_cpp = flmi_cpp.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                   stopIfNegativeGain=False, verbose=False)
    elapsed_cpp = time.perf_counter() - start_cpp
    print(f"C++ FLVMI Time: {elapsed_cpp:.4f} seconds")
    # PyTorch Implementation
    flmi_torch = FacilityLocationVariantMutualInformation(n=ground_data.shape[0],
                                                          num_queries=query_data.shape[0],
                                                          data=ground_data,
                                                          queryData=query_data,
                                                          metric="cosine")
    start_py = time.perf_counter()
    greedy_torch = flmi_torch.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                       stopIfNegativeGain=False, verbose=False)
    elapsed_torch = time.perf_counter() - start_py
    
    # Compare results
    indices_cpp = [item[0] for item in greedy_cpp]
    indices_torch = [item[0] for item in greedy_torch]

    gains_cpp = [item[1] for item in greedy_cpp]
    gains_torch = [item[1] for item in greedy_torch]

    indices_match = indices_cpp == indices_torch
    gains_match = all(abs(a - b) < 1e-5 for a, b in zip(gains_cpp, gains_torch))

    return indices_match, gains_match
    
def compare_logdetmi(ground_data, query_data, budget):
    # C++ Implementation
    logdetmi_cpp = LogDeterminantMutualInformationFunction(n=ground_data.shape[0],
                                                   num_queries=query_data.shape[0],
                                                   data=ground_data,
                                                   queryData=query_data,
                                                   metric="cosine",lambdaVal=1.0,magnificationEta=1)
    greedy_cpp = logdetmi_cpp.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                       stopIfNegativeGain=False, verbose=False)
    print("C++ Implementation Completed")
    print("PyTorch Implementation Starting")
    # PyTorch Implementation
    logdetmi_torch = LogDeterminantMutualInformation(n=ground_data.shape[0],
                                                     num_queries=query_data.shape[0],
                                                     data=ground_data,
                                                     queryData=query_data,
                                                     metric="cosine",lambdaVal=1.0,magnificationEta=1)
    greedy_torch = logdetmi_torch.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                           stopIfNegativeGain=False, verbose=False , show_progress=False,epsilon=1e-5,costs=None,costSensitiveGreedy=False)

    # Compare results
    print("C++ Implementation Results:")
    print(greedy_cpp)
    print("PyTorch Implementation Results:")
    print(greedy_torch)
    indices_cpp = [item[0] for item in greedy_cpp]
    indices_torch = [item[0] for item in greedy_torch]

    gains_cpp = [item[1] for item in greedy_cpp]
    gains_torch = [item[1] for item in greedy_torch]

    indices_match = indices_cpp == indices_torch
    gains_match = all(abs(a - b) < 1e-5 for a, b in zip(gains_cpp, gains_torch))

    return indices_match, gains_match

def compare_flcg(groundData , queryData , budget):
    # C++ Implementation
    flcg_cpp = FacilityLocationConditionalGainFunction(n=groundData.shape[0],
                                                      num_privates=queryData.shape[0],
                                                      data=groundData,
                                                      privateData=queryData,
                                                      metric="cosine")
                                    
    greedy_cpp = flcg_cpp.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                   stopIfNegativeGain=False, verbose=False)

    # PyTorch Implementation
    flcg_torch = FacilityLocationConditionalGain(n=groundData.shape[0],
                                                 num_privates=queryData.shape[0],
                                                 data=groundData,
                                                 privateData=queryData,
                                                 metric="cosine")
    greedy_torch = flcg_torch.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
                                       stopIfNegativeGain=False, verbose=False , show_progress=False,epsilon=1e-5,costs=None,costSensitiveGreedy=False)

    
    print("C++ Implementation Results:")
    print(greedy_cpp)
    print("PyTorch Implementation Results:")
    print(greedy_torch)

    indices_cpp = [item[0] for item in greedy_cpp]
    indices_torch = [item[0] for item in greedy_torch]

    gains_cpp = [item[1] for item in greedy_cpp]
    gains_torch = [item[1] for item in greedy_torch]

    indices_match = indices_cpp == indices_torch
    gains_match = all(abs(a - b) < 1e-5 for a, b in zip(gains_cpp, gains_torch))

    return indices_match, gains_match

if __name__ == "__main__":
    
    import torch
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    free , total = torch.cuda.mem_get_info()
    gb = 1024 ** 3
    print(f"Free memory: {free / gb:.2f} GB")
    print(f"Total memory: {total / gb:.2f} GB")
    groundData = torch.randn(50000,1024)
    singleQueryData = torch.randn(80,1024)
    budget = 500
    #130000
    """Compare C++ and PyTorch implementations of FacilityLocationVariantMutualInformation"""
    # indices_match, gains_match = compare_flmi(groundData, singleQueryData, budget)
    # print("\n")
    # print("Indices match:", indices_match) 
    # print("Gains match:", gains_match)
    
    """Compare C++ and PyTorch implementations of LOGDETMI"""
    
    # indices_match, gains_match = compare_logdetmi(groundData, singleQueryData, budget)
    # print("Indices match:", indices_match)
    # print("Gains match:", gains_match) 
    logdetmi_torch = LogDeterminantMutualInformation(n=groundData.shape[0],
                                                     num_queries=singleQueryData.shape[0],
                                                     data=groundData,
                                                     queryData=singleQueryData,
                                                     metric="cosine",lambdaVal=1.0,magnificationEta=1)
    free , total = torch.cuda.mem_get_info()
    gb = 1024 ** 3
    print(f"Free memory: {free / gb:.2f} GB")
    print(f"Total memory: {total / gb:.2f} GB")
    # greedy_torch = logdetmi_torch.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False,
    #                                        stopIfNegativeGain=False, verbose=False , show_progress=False,epsilon=1e-5,costs=None,costSensitiveGreedy=False)
    # print(greedy_torch)

    """Compare C++ and PyTorch implementations of FacilityLocationConditionalGain"""
    # indices_match, gains_match = compare_flcg(groundData, singleQueryData, budget)
    # print("\n")
    # print("Indices match:", indices_match)
    # print("Gains match:", gains_match)
