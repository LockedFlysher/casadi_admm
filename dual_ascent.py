import casadi

import optimization_problem

if __name__ == '__main__':
    x = casadi.SX.sym('x')
    y = casadi.SX.sym('y')

    # 创建实例
    op = optimization_problem.OptimizationProblem()
    op.add_objective_term(term=x*x)
    op.add_objective_term(term=y*y)
    print(op.get_objective_function())
    # 等式约束用的是mu，不等式约束用的lambda
    op.add_equality_constraint(x+y-1)
    op.add_inequality_constraint(x-y)
    print(op.get_multipliers())
    print(op.get_lagrange_function())
    history = op.dual_ascent(100)

    # 对偶问题的求解
