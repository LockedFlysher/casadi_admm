import casadi
from dual_ascent import *

if __name__ == '__main__':
    # note : 目前的x是必须加下划线的
    x1 = casadi.SX.sym('x1')
    x2 = casadi.SX.sym('x2')
    # 创建实例
    op = OptimizationProblem()
    op.set_objective_function(x1 * x1 + x2 * x2)
    op.set_variables([x1, x2])
    # 等式约束用的是mu，不等式约束用的lambda
    op.add_equality_constraint(x1 + x2 - 1)
    op.add_inequality_constraint(x1 - x2 + 0.5)
    result = op.dual_ascent(30, True)

    configuration = OptimizationProblemConfiguration(
        {"variables": [x1, x2],
         "objective_function": x1 ** 2 + x2 ** 2,
         # 等式约束要是线性的，满足 Ax = C
         "equality_constraints": {
             "A" : [ca.DM([1,1])],
             "B" : [ca.DM([1])]
         },
         "inequality_constraints": [x1 - x2 + 0.5],
         "initial_guess": [0, 0]
         })
    op2 = OptimizationProblem(configuration)
    result2 = op2.dual_ascent(30, False)

    print(result)
    print(result2)
