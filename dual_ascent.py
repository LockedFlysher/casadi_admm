import casadi

import optimization_problem

if __name__ == '__main__':
    x1 = casadi.SX.sym('x1')
    x2 = casadi.SX.sym('x2')

    # 创建实例
    op = optimization_problem.OptimizationProblem()
    op.add_objective_term(term=x1 * x1)
    op.add_objective_term(term=x2 * x2)
    print(op.get_objective_function())
    # 等式约束用的是mu，不等式约束用的lambda
    op.add_equality_constraint(x1 + x2 - 1)
    op.add_inequality_constraint(x1 - x2)
    print(op.get_multipliers())
    print(op.get_lagrange_function())

    # 对偶问题的求解
    history = op.dual_ascent(100)
