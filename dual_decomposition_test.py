# 导入 DualDecomposition 类
from dual_decomposition import DualDecomposition
from dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
import matplotlib.pyplot as plt


# 创建测试函数
def test_dual_decomposition():
    """
    测试对偶分解算法
    """
    print("开始测试对偶分解算法...")

    # 创建一个简单的优化问题：min f(x) = x1^2 + x2^2 + x3^2 + x4^2
    opt_problem = OptimizationProblem()

    # 定义变量
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')

    # 设置变量
    opt_problem.set_variables([x1, x2, x3, x4])

    # 设置目标函数
    objective = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2
    opt_problem.set_objective_function(objective)

    # 添加一些约束
    # x1 + x2 = 1
    opt_problem.add_equality_constraint(x1 + x2 - 1)
    # x3 + x4 = 2
    opt_problem.add_equality_constraint(x3 + x4 - 2)

    print("原始问题已创建")

    # 方法1：使用decompose_problem分解问题，然后手动配置子问题
    print("\n方法1：使用decompose_problem分解问题")
    dd_solver1 = DualDecomposition(opt_problem)
    subproblems = dd_solver1.decompose_problem([[0, 1], [2, 3]])

    # 手动配置子问题
    # 子问题1: min x1^2 + x2^2 s.t. x1 + x2 = 1
    subproblems[0].set_objective_function(x1 ** 2 + x2 ** 2)
    subproblems[0].add_equality_constraint(x1 + x2 - 1)

    # 子问题2: min x3^2 + x4^2 s.t. x3 + x4 = 2
    subproblems[1].set_objective_function(x3 ** 2 + x4 ** 2)
    subproblems[1].add_equality_constraint(x3 + x4 - 2)

    # 设置ADMM参数
    dd_solver1.set_rho(1.0)
    dd_solver1.set_alpha(0.5)

    # 生成ADMM函数
    dd_solver1.generate_admm_functions()
    print("ADMM函数已生成")

    # 求解问题
    max_iter = 100
    tol = 1e-6
    result1 = dd_solver1.solve(max_iter, tol)

    print("\n方法1求解结果:")

    # 方法2：使用OptimizationProblemConfiguration配置子问题
    print("\n方法2：使用OptimizationProblemConfiguration配置子问题")

    # 创建子问题1的配置
    subproblem1_config = OptimizationProblemConfiguration({
        'variables': [x1, x2],
        'objective_function': x1 ** 2 + x2 ** 2,
        'equality_constraints': [x1 + x2 - 1],
        'inequality_constraints': [],
        'initial_guess': ca.DM.zeros(2)
    })

    # 创建子问题2的配置
    subproblem2_config = OptimizationProblemConfiguration({
        'variables': [x3, x4],
        'objective_function': x3 ** 2 + x4 ** 2,
        'equality_constraints': [x3 + x4 - 2],
        'inequality_constraints': [],
        'initial_guess': ca.DM.zeros(2)
    })

    # 创建对偶分解求解器，直接使用配置
    dd_solver2 = DualDecomposition(opt_problem, [subproblem1_config, subproblem2_config])

    # 设置ADMM参数
    dd_solver2.set_rho(1.0)
    dd_solver2.set_alpha(0.5)

    # 生成ADMM函数
    dd_solver2.generate_admm_functions()

    # 求解问题
    result2 = dd_solver2.solve(max_iter, tol)

    print("\n方法2求解结果:")

    # 方法3：先创建求解器，然后添加配置好的子问题
    print("\n方法3：先创建求解器，然后添加配置好的子问题")
    dd_solver3 = DualDecomposition(opt_problem)

    # 添加子问题
    dd_solver3.add_subproblem_with_configuration(subproblem1_config)
    dd_solver3.add_subproblem_with_configuration(subproblem2_config)

    # 设置ADMM参数
    dd_solver3.set_rho(1.0)
    dd_solver3.set_alpha(0.5)

    # 生成ADMM函数
    dd_solver3.generate_admm_functions()

    # 求解问题
    result3 = dd_solver3.solve(max_iter, tol)

    print("\n方法3求解结果:")

    # 返回方法1的结果用于可能的可视化
    return result1


# 运行测试
if __name__ == "__main__":
    result = test_dual_decomposition()

