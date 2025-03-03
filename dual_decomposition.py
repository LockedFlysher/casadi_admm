from dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
from typing import Dict, List, Union, Optional, Any, Tuple


class DualDecomposition:
    """
    基于ADMM（交替方向乘子法）的对偶分解求解器
    """

    def __init__(self, optimization_problem: OptimizationProblem,
                 subproblem_configs: List[OptimizationProblemConfiguration] = None):
        """
        初始化对偶分解求解器，设置目标函数，变量，约束，是描述的总的一个优化问题，未被分解

        Args:
            optimization_problem: 优化问题实例
            subproblem_configs: 可选的子问题配置列表
        """
        self._opt_problem = optimization_problem

        # ADMM参数
        self._rho = 1.0  # 惩罚参数
        self._alpha = 1.0  # 松弛参数

        # 分解相关变量
        self._subproblems = []  # 子问题列表
        self._consensus_variables = []  # 一致性变量
        self._z = None  # 全局变量
        self._u = None  # 对偶变量
        self._local_xs = []  # 局部变量

        # 函数
        self._update_x_functions = []  # 更新局部变量的函数
        self._update_z_function = None  # 更新全局变量的函数
        self._update_u_function = None  # 更新对偶变量的函数

        # 状态标志
        self._decomposition_ready = False

        # 如果提供了子问题配置，则直接添加子问题
        if subproblem_configs:
            for config in subproblem_configs:
                self.add_subproblem_with_configuration(config)
            self._decomposition_ready = True

    def decompose_problem(self, decomposition_indices: List[List[int]]) -> List[OptimizationProblem]:
        """
        将原问题分解为多个子问题，仅创建子问题的框架，不分配目标函数和约束

        Args:
            decomposition_indices: 变量分解索引，每个子列表包含一个子问题的变量索引

        Returns:
            子问题列表，用于手动设置目标函数和约束
        """
        # 获取原问题的变量
        original_xs = self._opt_problem.xs

        # 创建子问题
        for indices in decomposition_indices:
            # 提取子问题的变量
            sub_xs = [original_xs[i] for i in indices]
            sub_x = ca.vertcat(*sub_xs)

            # 创建子问题实例
            subproblem = OptimizationProblem()
            subproblem.set_variables(sub_xs)

            # 添加到子问题列表
            self._subproblems.append(subproblem)
            self._local_xs.append(sub_x)

            # 创建对应的一致性变量和对偶变量
            self._consensus_variables.append(ca.SX.sym(f'z_{len(self._subproblems) - 1}', sub_x.size1()))

        # 初始化全局变量z和对偶变量u
        self._z = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]
        self._u = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]

        # 标记分解已完成
        self._decomposition_ready = True

        return self._subproblems

    def add_subproblem_with_configuration(self, config: OptimizationProblemConfiguration) -> OptimizationProblem:
        """
        使用OptimizationProblemConfiguration添加子问题

        Args:
            config: 子问题的配置

        Returns:
            新创建的子问题实例
        """
        # 创建子问题实例
        subproblem = OptimizationProblem(config)

        # 获取子问题的变量
        sub_xs = config.variables
        sub_x = ca.vertcat(*sub_xs)

        # 添加到子问题列表
        self._subproblems.append(subproblem)
        self._local_xs.append(sub_x)

        # 创建对应的一致性变量和对偶变量
        self._consensus_variables.append(ca.SX.sym(f'z_{len(self._subproblems) - 1}', sub_x.size1()))

        # 初始化全局变量z和对偶变量u
        self._z = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]
        self._u = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]

        # 标记分解已完成
        self._decomposition_ready = True

        return subproblem

    def add_configured_subproblem(self, subproblem: OptimizationProblem) -> None:
        """
        添加已配置好的子问题

        Args:
            subproblem: 已配置好的子问题实例
        """
        if not isinstance(subproblem, OptimizationProblem):
            raise TypeError("子问题必须是OptimizationProblem的实例")

        # 获取子问题的变量
        sub_xs = subproblem.xs

        # 添加到子问题列表
        self._subproblems.append(subproblem)
        self._local_xs.append(sub_xs)

        # 创建对应的一致性变量和对偶变量
        self._consensus_variables.append(ca.SX.sym(f'z_{len(self._subproblems) - 1}', sub_xs.size1()))

        # 初始化全局变量z和对偶变量u
        self._z = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]
        self._u = [ca.DM.zeros(var.size1()) for var in self._consensus_variables]

        # 标记分解已完成
        self._decomposition_ready = True

    def generate_admm_functions(self):
        """
        生成ADMM算法所需的函数
        """
        if not self._decomposition_ready:
            raise ValueError("请先调用decompose_problem方法")

        # 为每个子问题生成更新x的函数
        for i, subproblem in enumerate(self._subproblems):
            # 获取子问题的变量和约束
            x_i = self._local_xs[i]
            z_i = self._consensus_variables[i]
            # ui表示的是mu除以系数rho
            u_i = ca.SX.sym(f'u_{i}', x_i.size1())

            # 构建增广拉格朗日函数
            obj = subproblem.get_objective_expression()
            augmented_term = (self._rho / 2) * ca.sumsqr(x_i - z_i + u_i)
            augmented_obj = obj + augmented_term

            # 计算梯度
            grad = ca.gradient(augmented_obj, x_i)
            next_x = x_i - self._alpha * grad

            # 创建更新函数
            update_x_func = ca.Function(f'update_x_{i}', [x_i, z_i, u_i], [next_x])
            self._update_x_functions.append(update_x_func)

        # 生成更新z的函数
        z_sym = ca.vertcat(*self._consensus_variables)
        x_sym = ca.vertcat(*self._local_xs)
        u_sym = ca.vertcat(*[ca.SX.sym(f'u_{i}', x.size1()) for i, x in enumerate(self._local_xs)])

        # z更新：z = (1/N) * sum(x_i + u_i)
        next_z = x_sym + u_sym
        self._update_z_function = ca.Function('update_z', [x_sym, u_sym], [next_z])

        # u更新：u = u + (x - z)
        next_u = u_sym + (x_sym - z_sym)
        self._update_u_function = ca.Function('update_u', [x_sym, z_sym, u_sym], [next_u])

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> Dict[str, Any]:
        """
        使用ADMM算法求解分解后的问题

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            求解结果字典
        """
        if not self._decomposition_ready:
            raise ValueError("请先调用decompose_problem方法")

        # 确保ADMM函数已生成
        if len(self._update_x_functions) == 0:
            self.generate_admm_functions()

        # 初始化局部变量
        x = [ca.DM.zeros(x_i.size1()) for x_i in self._local_xs]

        # 初始化变量用于存储上一次迭代的z值
        primal_res = float('inf')
        dual_res = float('inf')
        k = 0

        # 迭代求解
        for k in range(max_iter):
            # 保存当前的z值，用于计算对偶残差
            z_old = [ca.DM(z_i) for z_i in self._z]

            # 1. 更新所有子问题的x
            for i in range(len(self._subproblems)):
                x[i] = self._update_x_functions[i](x[i], self._z[i], self._u[i])

            # 2. 更新全局变量z
            x_concat = ca.vertcat(*x)
            u_concat = ca.vertcat(*self._u)
            z_new = self._update_z_function(x_concat, u_concat)

            # 分解z回各个子问题
            z_split = ca.vertsplit(z_new)
            self._z = z_split

            # 3. 更新对偶变量u
            u_new = self._update_u_function(x_concat, z_new, u_concat)
            u_split = ca.vertsplit(u_new)
            self._u = u_split

            # 检查收敛性
            primal_res = ca.norm_2(ca.vertcat(*[x[i] - self._z[i] for i in range(len(x))]))

            # 确保z_old和self._z长度一致
            if len(z_old) == len(self._z):
                dual_res = ca.norm_2(ca.vertcat(*[self._z[i] - z_old[i] for i in range(len(self._z))]))
            else:
                # 如果长度不一致，可能是因为z_split的结构改变了
                dual_res = ca.norm_2(z_new - ca.vertcat(*z_old)) if len(z_old) > 0 else float('inf')

            if primal_res < tol and dual_res < tol:
                break

        # 返回结果
        result = {
            'x': x,
            'z': self._z,
            'u': self._u,
            'iterations': k + 1,
            'primal_residual': float(primal_res),
            'dual_residual': float(dual_res)
        }

        return result

    def set_rho(self, rho: float):
        """
        设置ADMM惩罚参数

        Args:
            rho: 惩罚参数值
        """
        self._rho = rho

    def set_alpha(self, alpha: float):
        """
        设置松弛参数

        Args:
            alpha: 松弛参数值
        """
        self._alpha = alpha

    def get_subproblems(self) -> List[OptimizationProblem]:
        """
        获取所有子问题

        Returns:
            子问题列表
        """
        if not self._decomposition_ready:
            raise ValueError("请先调用decompose_problem方法")

        return self._subproblems

    def get_subproblem(self, index: int) -> OptimizationProblem:
        """
        获取指定索引的子问题

        Args:
            index: 子问题索引

        Returns:
            子问题实例
        """
        if not self._decomposition_ready:
            raise ValueError("请先调用decompose_problem方法")

        if index < 0 or index >= len(self._subproblems):
            raise ValueError(f"子问题索引{index}超出范围")

        return self._subproblems[index]

