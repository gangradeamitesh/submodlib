
import numpy as np
import torch

def validate_n(n):
    if n <= 0:
        raise Exception("ERROR: Number of elements in ground set must be positive")

def validate_mode(mode):
    if mode not in ['dense', 'sparse', 'clustered']:
        raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")
		

def validate_sep_rep(separate_rep, mode , n_rep):
    if separate_rep == True:
        if n_rep  is None or n_rep <=0:
            raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
        if mode != "dense":
                raise Exception("Only dense mode supported if separate_rep = True")

"""Checking the spase or dense kernel"""
def validate_sijs(sijs_data_type, mode , num_neighbors , separate_rep):
    # if sijs_data_type == scipy.sparse.csr.csr_matrix:
    #     if num_neighbors is None or num_neighbors <= 0:
    #         raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
    #     if mode != "sparse":
    #         raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
    if sijs_data_type == torch.Tensor:
        if separate_rep is None:
            raise Exception("ERROR: separate_rep bool must be specified with custom dense kernel")
        if mode != "dense":
            raise Exception("ERROR: Dense kernel provided, but mode is not dense")
    else:
        raise Exception("Invalid kernel provided")