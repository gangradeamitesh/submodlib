from .base_optimizer import BaseOptimizer


class NaiveGreedy(BaseOptimizer):

    def __init__(self) -> None:
        pass

    def maximize(self ,function , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        output = []
        for _ in range(budget):
            print("Marginal Gain")
            print(type(function))
            gain = function.marginalGain(None, None)

        return None