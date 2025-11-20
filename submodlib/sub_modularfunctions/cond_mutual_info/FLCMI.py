from ..userValidator import validate_n, validate_mode, validate_sep_rep, validate_sijs
from ..cal_simi_kernel import DenseSimilarity
from ..base_function import BaseFunction
from ..optimizers.optimizer_factory import OptimizerFactory
import torch
import numpy as np
from sub_modularfunctions.conditional_gain_functions import FLCG
from 


class FacilityLocationConditionalMutualInfomation(BaseFunction):

    def __init__(self, n = None, num_queries=None, data_sijs=None, query_sijs=None, private_sijs=None, data=None, queryData=None
                 privateData = None, metric="cosine",magnificationEta=1, privacyHardness=1):
        self.FLCG = FLCG()