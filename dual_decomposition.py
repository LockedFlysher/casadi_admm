from optimization_problem import OptimizationProblem
from optimization_problem import OptimizationProblemConfiguration
import casadi as ca

class SubOptimizationProblemConfiguration:
    def __init__(self, configuration):
        self.variables = configuration['variables']
        self.objective_function = configuration['objective_function']

class DualDecompositionProblem:
    def __init__(self):
        # 主要的目的也是要把优化问题对象给利用起来，完成分布式的更新
        self.__optimization_problem__ = []
        self.__whole_problem__ = None
        # 使用DM成数值上的求解，我们先更新x，再更新mu和lambda
        self.__total_x__ = None
        self.__each_mu_size__ = []
        self.__total_mu__ = None
        self.__each_lambda_size__ = []
        self.__total_lambda__ = None
        # A矩阵的大小是 约束数量x变量数量
        self.__each_A__ = []
        self.__A__ = None
        self.__each_B__ = []
        self.__B__ = None

    def set_whole_problem(self,configuration:OptimizationProblemConfiguration):
        self.__total_x__ = configuration.variables
        self.__total_mu__ = len(configuration.equality_constraints)
        # 分配一下各种矩阵的大小
        self.__A__ = ca.DM.zeros(len(configuration.equality_constraints),len(configuration.variables))
        self.__B__ = ca.DM.zeros(len(configuration.inequality_constraints),len(configuration.variables))
        #

    def add_sub_problem(self, configuration:SubOptimizationProblemConfiguration):
        sub_optimal_ OptimizationProblemConfiguration()
        self.__optimization_problem__.append(configuration)

    # 给子问题的lambda和mu进行赋值，但是我们上面需要一个管理器，OptimizationProblemConfiguration是一个很好的工具
    def grant_mu_and_lambda(self):

        pass

    def compute_sub_problems(self):

        for optimization_problem in self.__optimization_problem__:
            optimization_problem.compute_next_x()

