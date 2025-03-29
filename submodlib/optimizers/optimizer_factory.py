
from naive_greedy import NaiveGreedy
from stochastic_greedy import StochasticGreedy
from lazy_greedy import LazyGreedy
from lazier_greedy import LazierThanLazyGreedy
from utils.contants import Constant

class OptimizerFactory:
    def get_optimizer(self , optimizer):
        if optimizer==Constant.naive_greedy:
            return NaiveGreedy()
        elif optimizer==Constant.stocastic_greedy:
            return StochasticGreedy()
        elif optimizer==Constant.lazy_greedy:
            return LazyGreedy()
        elif optimizer==Constant.lazier_greedy:
            return LazierThanLazyGreedy()
        else:
            raise Exception("Invalid optimizer. Can be {Constant.naive_greedy}, {Constant.stocastic_greedy},  and ‘LazierThanLazyGreedy’.")