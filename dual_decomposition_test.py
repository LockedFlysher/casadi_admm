from fontTools.misc.cython import returns

import dual_decomposition
from dual_decomposition import DualDecompositionProblem
from optimization_problem import OptimizationProblemConfiguration
import casadi as ca

if __name__ == '__main__':
    dp = DualDecompositionProblem()
    # note : 目前的x是必须加下划线的
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')

    configuration = OptimizationProblemConfiguration(
        {"variables": [x1, x2],
         "objective_function": x1 ** 2 + x2 ** 2,
         "equality_constraints": [x1 + x2 - 1],
         "inequality_constraints": [x1 - x2],
         "initial_guess": [0, 0]
         })

    sub_configuration1 = OptimizationProblemConfiguration(
        {"variables": [x1],
         "objective_function": x1 ** 2,
         "initial_guess": [0]
         })

    sub_configuration2 = OptimizationProblemConfiguration(
        {"variables": [x2],
         "objective_function": x2 ** 2,
         "initial_guess": [0]
         })

    # 1.添加对偶问题的子问题
    dp.set_whole_problem(configuration=configuration)

    pass