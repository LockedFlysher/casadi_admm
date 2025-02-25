import casadi
import optimization_problem
from optimization_problem import OptimizationProblemConfiguration

if __name__ == '__main__':
    # note : 目前的x是必须加下划线的
    x1 = casadi.SX.sym('x1')
    x2 = casadi.SX.sym('x2')
    # 创建实例
    op = optimization_problem.OptimizationProblem()
    op.set_objective_function(x1 * x1 + x2 * x2)
    op.set_variables([x1, x2])
    # 等式约束用的是mu，不等式约束用的lambda
    op.add_equality_constraint(x1 + x2 - 1)
    op.add_inequality_constraint(x1 - x2)
    result = op.dual_ascent(5, True)

    configuration = OptimizationProblemConfiguration(
        {"variables": [x1, x2],
         "objective_function": x1 ** 2 + x2 ** 2,
         "equality_constraints": [x1 + x2 - 1],
         "inequality_constraints": [x1 - x2],
         "initial_guess": [0, 0]
         })
    op2 = optimization_problem.OptimizationProblem(configuration)
    result2 = op2.dual_ascent(5, False)

    print(result)
    print(result2)
