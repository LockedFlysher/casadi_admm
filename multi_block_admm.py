from casadi import vertcat
from sympy.physics.vector import gradient

from dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
from typing import Dict, List, Union, Optional, Any, Tuple

class MultiBlockADMM(OptimizationProblem):
    """
    基于ADMM（交替方向乘子法）的多块分解求解器
    解决问题：min sum(f_i(x_i)) s.t. sum(A_i*x_i) = c
    """

    def __init__(self):
        """
        初始化多块ADMM求解器
        """
        # ADMM参数
        self._rho = 1.0  # 惩罚参数

        # 初始化Function需要用的矩阵
        self.A = []
        self.C = []
        self.U = []

        # 分解相关变量
        self._subproblems = []  # 子问题列表
        self._u = None  # 对偶变量，就是y/rho
        self._x = None

        # 更新x的函数是有必要拿上来的
        self._update_x_functions = []  # 更新各子问题x的函数列表
        self._update_u_function = None  # 更新对偶变量的函数

        # 收敛历史
        self._primal_residuals = []
        self._dual_residuals = []

        # 状态标志
        self._problem_configured = False

    def add_subproblem(self, config: OptimizationProblemConfiguration):
        """
        添加子问题

        Args:
            config: 子问题的配置
            A: 约束矩阵A_i，如果为None则默认为单位矩阵
        """
        # 创建子问题实例
        subproblem = OptimizationProblem(config)
        # 设置约束矩阵
        # 添加到列表
        self._subproblems.append(subproblem)

        return len(self._subproblems) - 1  # 返回子问题索引

    def configure_problem(self, c=None):
        """
        配置多块ADMM问题：min sum(f_i(x_i)) s.t. sum(A_i*x_i) = c

        Args:
            c: 约束常数c，如果为None则默认为零向量
        """
        if len(self._subproblems) == 0:
            raise ValueError("请先添加子问题")

        # 确定约束维度（使用第一个A矩阵的行数）
        constraint_dim = self._A_matrices[0].shape[0]

        # 检查所有A矩阵的行数是否一致
        for i, A_i in enumerate(self._A_matrices):
            if A_i.shape[0] != constraint_dim:
                raise ValueError(f"约束矩阵A_{i}的行数与A_0不一致")

        # 设置约束常数c
        if c is None:
            self._C_matrices = ca.DM.zeros(constraint_dim)
        else:
            self._C_matrices = c

        # 初始化对偶变量u
        self._u = ca.DM.zeros(constraint_dim)

        # 标记问题已配置
        self._problem_configured = True

    def generate_admm_functions(self):
        """
        生成多块ADMM算法所需的函数
        """
        if not self._problem_configured:
            raise ValueError("请先调用configure_problem方法")

        # 1. 尝试轮番更新x
        augmented_lagraunge_function = ca.SX.zeros(1)
        residual_vector = []
        for i,subproblem in enumerate(self._subproblems):
            augmented_lagraunge_function += self._rho/2 * subproblem.get_objective_expression()
            self.A.append(subproblem.A)
            self.C.append(subproblem.C)
            # 残差项的计算
            residual_vector.append(subproblem.A @ subproblem.get_xs() - subproblem.C)
        residual = ca.vertcat(*residual_vector)
        augmented_lagraunge_function += (residual+self._u).T*(residual+self._u)

        # 已知： 所有的变量的当前值，已知，求在此点的子问题的变量的梯度，更新的是子问题被分离的变量中的一组向量
        # todo : 用gradient求出梯度表达式，得到方程
        gradient = ca.gradient(augmented_lagraunge_function, self._x)


        # 为每个子问题生成更新x的函数
        for i, subproblem in enumerate(self._subproblems):
            # 获取子问题的变量
            x_i = ca.SX.sym(f'x_{i}', self._x_vars[i].size1())

            # 创建其他子问题变量的符号
            other_x_syms = []
            for j in range(len(self._subproblems)):
                if j != i:
                    other_x_syms.append(ca.SX.sym(f'x_{j}', self._x_vars[j].size1()))

            # 创建对偶变量符号
            u_sym = ca.SX.sym('u', self._u.size1())

            # 获取子问题的目标函数
            f_i = subproblem.get_objective_expression()

            # 构建增广拉格朗日函数
            # L_ρ(x_i, x_{-i}, u) = f_i(x_i) + (ρ/2)||sum(A_j*x_j) - c + u||_2^2
            term1 = self._A_matrices[i] @ x_i

            # 添加其他子问题的贡献
            other_idx = 0
            for j in range(len(self._subproblems)):
                if j != i:
                    term1 = term1 + self._A_matrices[j] @ other_x_syms[other_idx]
                    other_idx += 1

            augmented_term = (self._rho / 2) * ca.sumsqr(term1 - self._C_matricies + u_sym)
            augmented_obj = f_i + augmented_term

            # 计算梯度
            grad = ca.gradient(augmented_obj, x_i)

            # 创建更新函数
            # 注意：这里使用简化的梯度下降步骤，实际应用中可能需要求解优化子问题
            next_x = x_i - grad

            # 函数输入：当前子问题的x，其他子问题的x，对偶变量u
            inputs = [x_i] + other_x_syms + [u_sym]
            self._update_x_functions.append(ca.Function(f'update_x_{i}', inputs, [next_x]))

        # 生成更新u的函数
        # 创建所有子问题变量的符号
        x_syms = [ca.SX.sym(f'x_{i}', self._x_vars[i].size1()) for i in range(len(self._subproblems))]
        u_sym = ca.SX.sym('u', self._u.size1())

        # 计算约束表达式：sum(A_i*x_i) - c
        constraint_expr = ca.DM.zeros(self._C_matricies.shape)
        for i in range(len(self._subproblems)):
            constraint_expr = constraint_expr + self._A_matrices[i] @ x_syms[i]
        constraint_expr = constraint_expr - self._C_matricies

        # u更新：u = u + (sum(A_i*x_i) - c)
        next_u = u_sym + constraint_expr

        # 函数输入：所有子问题的x，对偶变量u
        self._update_u_function = ca.Function('update_u', x_syms + [u_sym], [next_u])

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> Dict[str, Any]:
        """
        使用多块ADMM算法求解问题

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            求解结果字典
        """
        if not self._problem_configured:
            raise ValueError("请先调用configure_problem方法")

        # 确保ADMM函数已生成
        if len(self._update_x_functions) == 0:
            self.generate_admm_functions()

        # 初始化变量
        x_values = [ca.DM.zeros(x_i.size1()) for x_i in self._x_vars]
        u = ca.DM.zeros(self._u.size1())

        # 初始化收敛历史
        self._primal_residuals = []
        self._dual_residuals = []

        # 迭代求解
        primal_res = float('inf')
        dual_res = float('inf')
        k = 0

        for k in range(max_iter):
            # 保存当前的x值，用于计算对偶残差
            x_old = [ca.DM(x_i) for x_i in x_values]

            # 1. 依次更新每个子问题的x
            for i in range(len(self._subproblems)):
                # 准备函数输入：当前子问题的x，其他子问题的x，对偶变量u
                inputs = [x_values[i]]
                for j in range(len(self._subproblems)):
                    if j != i:
                        inputs.append(x_values[j])
                inputs.append(u)

                # 更新x_i
                x_values[i] = self._update_x_functions[i](*inputs)

            # 2. 更新对偶变量u
            u = self._update_u_function(*(x_values + [u]))

            # 3. 计算残差
            # 原始残差：||sum(A_i*x_i) - c||
            constraint_value = ca.DM.zeros(self._C_matricies.shape)
            for i in range(len(self._subproblems)):
                constraint_value = constraint_value + self._A_matrices[i] @ x_values[i]
            primal_res = ca.norm_2(constraint_value - self._C_matricies)

            # 对偶残差：||rho * sum(A_i^T * (x_i - x_i_old))||
            dual_term = ca.DM.zeros(u.shape)
            for i in range(len(self._subproblems)):
                dual_term = dual_term + self._A_matrices[i].T @ (x_values[i] - x_old[i])
            dual_res = self._rho * ca.norm_2(dual_term)

            # 记录收敛历史
            self._primal_residuals.append(float(primal_res))
            self._dual_residuals.append(float(dual_res))

            # 检查收敛性
            if primal_res < tol and dual_res < tol:
                break

        # 返回结果
        result = {
            'x': x_values,
            'u': u,
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

    def get_subproblem(self, index: int) -> OptimizationProblem:
        """
        获取指定索引的子问题

        Args:
            index: 子问题索引

        Returns:
            子问题实例
        """
        if index < 0 or index >= len(self._subproblems):
            raise ValueError(f"子问题索引{index}超出范围")
        return self._subproblems[index]

    def get_subproblems(self) -> List[OptimizationProblem]:
        """
        获取所有子问题

        Returns:
            子问题列表
        """
        return self._subproblems

    def get_convergence_history(self) -> Dict[str, List[float]]:
        """
        获取收敛历史

        Returns:
            包含原始残差和对偶残差历史的字典
        """
        return {
            'primal_residuals': self._primal_residuals,
            'dual_residuals': self._dual_residuals
        }
