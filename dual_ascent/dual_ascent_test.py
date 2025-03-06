from dual_ascent import *
import casadi as ca
import numpy as np
def test_optimization_problems():
    # 创建变量
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')

    print("========== 测试样例 1: 基本二次规划 ==========")
    # 最小化 x1^2 + x2^2，约束 x1 + x2 = 1
    config1 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": [[1, 1]], "B": [1]},
        initial_guess=[0, 0]
    )
    op1 = OptimizationProblem(config1)
    result1 = op1.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5]")
    print(f"实际结果: {result1}")

    print("\n========== 测试样例 2: 多约束问题 ==========")
    # 最小化 x1^2 + x2^2，约束 x1 + x2 = 1, x1 - x2 = 0
    config2 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={
            "A": [[1, 1], [1, -1]],  # 两个约束行
            "B": [1, 0]
        },
        initial_guess=[0, 0]
    )
    op2 = OptimizationProblem(config2)
    result2 = op2.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5]")
    print(f"实际结果: {result2}")

    print("\n========== 测试样例 3: 三维问题 ==========")
    # 最小化 x1^2 + x2^2 + x3^2，约束 x1 + x2 + x3 = 1
    config3 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3],
        objective_function=x1 ** 2 + x2 ** 2 + x3 ** 2,
        equality_constraints={"A": [[1, 1, 1]], "B": [1]},
        initial_guess=[0, 0, 0]
    )
    op3 = OptimizationProblem(config3)
    result3 = op3.dual_ascent(1000, False)
    print(f"期望结果: [1/3, 1/3, 1/3]")
    print(f"实际结果: {result3}")

    print("\n========== 测试样例 4: 线性目标函数 ==========")
    # 最小化 2*x1 + 3*x2，约束 x1 + x2 = 1, x1 >= 0, x2 >= 0
    config4 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=2 * x1 + 3 * x2,
        equality_constraints={"A": [[1, 1]], "B": [1]},
        inequality_constraints=[0 - x1, 0 - x2]  # x1 >= 0, x2 >= 0
    )
    op4 = OptimizationProblem(config4)
    result4 = op4.dual_ascent(1000, False)
    print(f"期望结果: [1, 0]")
    print(f"实际结果: {result4}")

    print("\n========== 测试样例 5: numpy格式约束 ==========")
    # 最小化 x1^2 + x2^2，约束 x1 + x2 = 1 (使用numpy格式)
    A_np = np.array([[1, 1]])
    B_np = np.array([1])
    config5 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": A_np, "B": B_np}
    )
    op5 = OptimizationProblem(config5)
    result5 = op5.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5]")
    print(f"实际结果: {result5}")

    print("\n========== 测试样例 6: casadi DM格式 ==========")
    # 最小化 x1^2 + x2^2，约束 x1 + x2 = 1 (使用casadi DM格式)
    A_dm = ca.DM([[1, 1]])
    B_dm = ca.DM([1])
    config6 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": A_dm, "B": B_dm},
        initial_guess=ca.DM([0.1, 0.1])  # 使用casadi DM作为初始猜测
    )
    op6 = OptimizationProblem(config6)
    result6 = op6.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5]")
    print(f"实际结果: {result6}")

    print("\n========== 测试样例 7: 四维问题 ==========")
    # 最小化 x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2，约束 x1 + x2 + x3 + x4 = 1
    config7 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3, x4],
        objective_function=x1 ** 2 + 2 * x2 ** 2 + 3 * x3 ** 2 + 4 * x4 ** 2,
        equality_constraints={"A": [[1, 1, 1, 1]], "B": [1]}
    )
    op7 = OptimizationProblem(config7)
    result7 = op7.dual_ascent(1000, False)
    print(f"期望结果: [大约 0.48, 0.24, 0.16, 0.12]")
    print(f"实际结果: {result7}")

    print("\n========== 测试样例 8: 混合约束表示 ==========")
    # 最小化 x1^2 + x2^2，使用不同格式的约束
    config8 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={
            "A": [np.array([1, 1]), ca.DM([2, 1])],  # 混合numpy和casadi
            "B": [1, 3]  # 列表
        }
    )
    op8 = OptimizationProblem(config8)
    result8 = op8.dual_ascent(1000, False)
    print(f"期望结果: [2.0, -1.0]")
    print(f"实际结果: {result8}")

    print("\n========== 测试样例 9: 转置向量作为约束 ==========")
    # 最小化 x1^2 + x2^2，使用转置向量表示约束
    A_col = ca.DM([1, 1]).T  # 创建行向量
    config9 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": A_col, "B": 1}
    )
    op9 = OptimizationProblem(config9)
    result9 = op9.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5]")
    print(f"实际结果: {result9}")

    print("\n========== 测试样例 10: 多个单独的矩阵约束 ==========")
    # 最小化 x1^2 + x2^2 + x3^2，有多个约束
    # 约束: x1 + x2 = 1, x2 + x3 = 1
    A1 = np.array([[1, 1, 0]])  # 第一个约束
    A2 = np.array([[0, 1, 1]])  # 第二个约束
    config10 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3],
        objective_function=x1 ** 2 + x2 ** 2 + x3 ** 2,
        equality_constraints={
            "A": [A1, A2],
            "B": [1, 1]
        },
        initial_guess=[0.1, 0.1, 0.1]
    )
    op10 = OptimizationProblem(config10)
    result10 = op10.dual_ascent(1000, False)
    print(f"期望结果: [0.5, 0.5, 0.5]")
    print(f"实际结果: {result10}")


def test_format_conversions():
    # 创建变量
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')

    print("\n========== 测试样例 16: 单行列表转换 ==========")
    # 单行列表表示约束
    config16 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3],
        objective_function=x1 ** 2 + x2 ** 2 + x3 ** 2,
        equality_constraints={"A": [1, 2, 3], "B": 6}
    )
    op16 = OptimizationProblem(config16)
    result16 = op16.dual_ascent(1000, False)
    print(f"期望结果: [1, 2, 3] 的某个比例")
    print(f"实际结果: {result16}")

    print("\n========== 测试样例 17: 复杂混合约束 ==========")
    # A是列表，B是numpy数组和标量的混合
    A_list = [[1, 1, 0], np.array([0, 1, 1])]
    B_mix = [np.array(1), 2.0]
    config17 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3],
        objective_function=x1 ** 2 + x2 ** 2 + x3 ** 2,
        equality_constraints={"A": A_list, "B": B_mix}
    )
    op17 = OptimizationProblem(config17)
    result17 = op17.dual_ascent(1000, False)
    print(f"期望结果: 满足约束的最优解")
    print(f"实际结果: {result17}")

    print("\n========== 测试样例 18: 稀疏约束矩阵 ==========")
    # 创建一个稀疏的约束矩阵（大部分元素为0）
    A_sparse = [[1, 0, 0], [0, 0, 1]]  # x1 = 2, x3 = 3
    B_sparse = [2, 3]
    config18 = OptimizationProblemConfiguration(
        variables=[x1, x2, x3],
        objective_function=x1 ** 2 + x2 ** 2 + x3 ** 2,
        equality_constraints={"A": A_sparse, "B": B_sparse}
    )
    op18 = OptimizationProblem(config18)
    result18 = op18.dual_ascent(1000, False)
    print(f"期望结果: [2, 0, 3]")
    print(f"实际结果: {result18}")

    print("\n========== 测试样例 19: 嵌套列表格式 ==========")
    # A是深度嵌套的列表
    A_nested = [[[1, 1]]]  # 额外嵌套一层
    config19 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": A_nested, "B": [1]}
    )
    op19 = OptimizationProblem(config19)
    result19 = op19.dual_ascent(1000, False)
    print(f"期望结果: 约为 [0.5, 0.5] 或报错")
    print(f"实际结果: {result19}")

    print("\n========== 测试样例 20: 空约束列表处理 ==========")
    # 传入空的约束列表
    config20 = OptimizationProblemConfiguration(
        variables=[x1, x2],
        objective_function=x1 ** 2 + x2 ** 2,
        equality_constraints={"A": [], "B": []}
    )
    op20 = OptimizationProblem(config20)
    result20 = op20.dual_ascent(1000, False)
    print(f"期望结果: [0, 0]")
    print(f"实际结果: {result20}")

if __name__ == '__main__':
    test_optimization_problems()
    test_format_conversions()

