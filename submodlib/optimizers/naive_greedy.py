from base_optimizer import BaseOptimizer
class NaiveGreedy(BaseOptimizer):

    def __init__(self) -> None:
        pass

    def maximize(self ,function , optimizer , budget , stopIfZeroGain , stopIfNegativeGain , epsilon , verbose , show_progress , costs , costSensitiveGreedy):
        """Maximize the function using the optimizer"""
        function.marginalGain(None, None)
        return None