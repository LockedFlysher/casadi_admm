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