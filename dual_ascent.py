import casadi

import optimization_problem

if __name__ == '__main__':
    # note : 目前的x是必须加下划线的
    x1 = casadi.SX.sym('x_1')
    x2 = casadi.SX.sym('x_2')
    # 创建实例
    op = optimization_problem.OptimizationProblem()
    op.set_objective_function(x1*x1+x2*x2)
    op.define_variables(x1,x2)
    # 等式约束用的是mu，不等式约束用的lambda
    op.add_equality_constraint(x1 + x2 - 1)
    op.add_inequality_constraint(x1 - x2)
    result = op.dual_ascent(5,True)
    print(result)
