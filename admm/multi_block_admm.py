from numpy.f2py.auxfuncs import throw_error
import matplotlib.pyplot as plt
from dual_ascent.dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
from typing import Dict, List, Any
import numpy as np


class MultiBlockADMM():
    """
    基于ADMM（交替方向乘子法）的多块分解求解器
    解决问题：min sum(f_i(x_i)) s.t. sum(A_i*x_i) = c
    """

    def __init__(self):
        """
        初始化多块ADMM求解器
        """
        # ADMM参数
        self._rho = 0.1  # 惩罚参数
        self._alpha = 0.1  # 梯度步长参数
        self._adaptive_rho = True  # 是否使用自适应惩罚参数
        self._verbose = True  # 是否输出详细信息

        self.Xk = None
        # ADMM管理整个问题的约束协调的缩放后的拉格朗日变量
        self.U_list = []  # 对偶变量，就是y/rho，这个是ADMM管理器2需要更新的参数
        self.U_sym = None  # 对偶变量，就是y/rho，这个是ADMM管理器2需要更新的参数
        self.Uk = None
        # 更新x的函数，用到上层的U对其进行更新，总共需要的函数是有 【约束的组数 + 1】 个，达成并行的更新
        self._update_x_functions = []  # 更新各子问题x的函数列表
        self._update_u_function = None  # 更新对偶变量的函数

        # 子问题列表内可以访问子问题的矩阵A和B，用来计算原问题的残差，更新变量和乘子的时候都要使用到
        self._subproblems = []  # 子问题列表
        self.augmented_lagrange_function = ca.SX.zeros(1)

        # 收敛的评判标准有两个，一个是原问题的残差[Ax-c]^T [Ax-c]、对偶的是对每一组变量求梯度，梯度要接近于0才对
        self._primal_residuals = []
        self._dual_residuals = []

    def add_subproblem(self, config: OptimizationProblemConfiguration):
        """
        添加子问题

        Args:
            config: 子问题的配置
        """
        subproblem = OptimizationProblem(config)
        self._subproblems.append(subproblem)

    def generate_admm_functions(self):
        """
        生成多块ADMM算法所需的函数 - 使用梯度下降方法（保留原有的封闭形式解）
        """
        # 子问题在设计的时候，需要保持子问题的目标函数的加和是和总问题一致的
        subproblem_xs_sym = []
        for subproblem in self._subproblems:
            subproblem_xs_sym.append(subproblem.get_xs())

        # 1.首要目标是求拉格朗日函数的符号表达式、残差的符号表达式，后续的更新都是靠他们
        self.U_list = []
        self.augmented_lagrange_function = ca.SX.zeros(1)
        residual_vector = []

        for i, subproblem in enumerate(self._subproblems):
            # 原始的目标函数的求和
            self.augmented_lagrange_function += subproblem.get_objective_expression()
            self.U_list.append(ca.SX.sym(f'u_{i}', subproblem.A.size1()))
            # 残差项的计算
            residual_vector.append(subproblem.A @ subproblem.get_xs() - subproblem.B)

        # 残差是向量，是和约束的数量一致的，U是协调变量，U的元素的数量是和约束的数量相等的
        self.U_sym = ca.vertcat(*self.U_list)
        residual = ca.vertcat(*residual_vector)
        self.augmented_lagrange_function += self._rho / 2 * ca.mtimes((residual + self.U_sym).T,
                                                                      (residual + self.U_sym))

        # 2.建立子问题的x的梯度的公式，通过一次梯度更新可以得到更新后的x的值，导出公式，放到Function的列表里
        self._update_x_functions = []  # 清空之前的函数

        for i, subproblem in enumerate(self._subproblems):
            # 计算梯度
            gradient = ca.gradient(self.augmented_lagrange_function, subproblem.get_xs())
            # 使用步长参数进行更新
            next_subproblem_x = subproblem.get_xs() - self._alpha * gradient

            # 创建更新函数
            subproblem_x_update_function = ca.Function(
                f"next_x_function_{i}",
                [self.U_sym, *subproblem_xs_sym],
                [next_subproblem_x]
            )

            if self._verbose:
                print(f"第 {i} 个x更新方程建立，输入为Uk、当前的Xk，输出为当前子问题的X_i_k+1")

            self._update_x_functions.append(subproblem_x_update_function)

        # 3.建立对偶变量更新函数
        next_whole_problem_u = self.U_sym + residual
        self._update_u_function = ca.Function(
            "next_u_function",
            [self.U_sym, *subproblem_xs_sym],
            [next_whole_problem_u]
        )

        if self._verbose:
            print("U更新方程建立，输入为Uk、经过更新后的Xk+1，输出为U_{k+1}")

    def solve(self, max_iter: int = 1000, tol: float = 1e-4) -> Dict[str, Any]:
        """
        使用多块ADMM算法求解问题

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            求解结果字典
        """
        # 检查求解器准备情况
        self.check()

        # 初始化变量
        self.Xk = []
        self.Uk = ca.DM.zeros(self.U_sym.size1())
        self._primal_residuals = []
        self._dual_residuals = []

        for i, subproblem in enumerate(self._subproblems):
            self.Xk.append(ca.DM(subproblem.get_initial_guess()))

        # 迭代求解
        for iterator in range(max_iter):
            # 保存旧值用于计算残差
            x_old = [x.full().copy() for x in self.Xk]
            u_old = self.Uk.full().copy()

            # 更新每个子问题的变量（使用梯度下降）
            for i, subproblem in enumerate(self._subproblems):
                self.Xk[i] = self._update_x_functions[i](self.Uk, *self.Xk)

            # 更新对偶变量
            self.Uk = self._update_u_function(self.Uk, *self.Xk)

            # 计算残差
            primal_res = self._calculate_primal_residual()
            dual_res = self._calculate_dual_residual(x_old)

            self._primal_residuals.append(primal_res)
            self._dual_residuals.append(dual_res)

            # 打印收敛信息
            if self._verbose and iterator % 10 == 0:
                print(
                    f"迭代 {iterator}: 原问题残差 = {primal_res:.6e}, 对偶残差 = {dual_res:.6e}, rho = {self._rho}, alpha = {self._alpha}")

            # 检查收敛
            if primal_res < tol and dual_res < tol:
                if self._verbose:
                    print(f"ADMM收敛于第 {iterator} 次迭代")
                break

            # 自适应调整参数
            if self._adaptive_rho and iterator > 0 and iterator % 5 == 0:
                self._update_parameters(primal_res, dual_res, iterator)

            # 如果残差变得过大，说明可能发散了
            if primal_res > 1e10 or dual_res > 1e10:
                if self._verbose:
                    print("警告: 残差过大，算法可能发散。尝试减小步长参数alpha和增大惩罚参数rho。")
                self._alpha *= 0.5  # 减小步长
                self._rho *= 2.0  # 增大惩罚参数
                break

        if iterator == max_iter - 1 and self._verbose:
            print(f"ADMM在达到最大迭代次数 {max_iter} 后停止")

        # 构建结果
        result = {
            'x': [x.full() for x in self.Xk],
            'u': self.Uk.full(),
            'iterations': iterator + 1,
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'convergence_history': self.get_convergence_history()
        }

        return result

    def _calculate_primal_residual(self):
        """计算原问题残差"""
        residual = ca.DM.zeros(1)
        for i, subproblem in enumerate(self._subproblems):
            r_i = subproblem.A @ self.Xk[i] - subproblem.B
            residual += ca.norm_2(r_i)
        return float(residual)

    def _calculate_dual_residual(self, x_old):
        """计算对偶残差"""
        dual_res = ca.DM.zeros(1)
        for i, subproblem in enumerate(self._subproblems):
            # 计算 x_i^(k+1) - x_i^k 的变化量
            diff = self.Xk[i] - x_old[i]
            dual_res += self._rho * ca.norm_2(subproblem.A @ diff)
        return float(dual_res)

    def _update_parameters(self, primal_res, dual_res, iteration):
        """自适应调整惩罚参数rho和步长alpha"""
        # 调整rho
        old_rho = self._rho

        if primal_res > 10 * dual_res:
            self._rho *= 1.5
        elif dual_res > 10 * primal_res:
            self._rho *= 0.8

        # 如果rho改变了，需要调整U以保持lambda = rho * U不变
        if old_rho != self._rho:
            self.Uk = (old_rho / self._rho) * self.Uk
            if self._verbose:
                print(f"调整惩罚参数: rho从 {old_rho} 更新到 {self._rho}")

        # 动态调整步长alpha（基于残差）
        old_alpha = self._alpha
        if iteration > 20:  # 前几次迭代不调整
            if len(self._primal_residuals) >= 3:
                # 如果残差持续增加，减小步长
                if (self._primal_residuals[-1] > self._primal_residuals[-2] and
                        self._primal_residuals[-2] > self._primal_residuals[-3]):
                    self._alpha *= 0.8
                # 如果残差持续减少，可以尝试增大步长
                elif (self._primal_residuals[-1] < self._primal_residuals[-2] and
                      self._primal_residuals[-2] < self._primal_residuals[-3]):
                    self._alpha *= 1.1
                    self._alpha = min(self._alpha, 0.5)  # 步长上限

            if old_alpha != self._alpha and self._verbose:
                print(f"调整步长参数: alpha从 {old_alpha} 更新到 {self._alpha}")

    def check(self):
        """检查ADMM求解器的设置是否合法"""
        # 检查子问题和函数是否已添加
        if len(self._subproblems) == 0:
            raise ValueError("未添加任何子问题")

        if len(self._update_x_functions) == 0:
            raise ValueError("尚未生成ADMM函数，请先调用generate_admm_functions()")

        # 检查参数设置
        if self._rho <= 0:
            raise ValueError("惩罚参数rho必须为正数")

        if self._alpha <= 0:
            raise ValueError("步长参数alpha必须为正数")

        # 检查拉格朗日乘子维度
        if self._verbose:
            print(f"检查项1: 拉格朗日乘子维度是 {self.U_sym.size1()}")

        lagrange_multiplier_counter = 0
        for i, subproblem in enumerate(self._subproblems):
            if self._verbose:
                print(f"正在检查编号为【{i}】的子问题")

            # 检查维度是否与函数和拉格朗日匹配
            if self._verbose:
                print(f"线性约束矩阵A: {subproblem.A.size1()}×{subproblem.A.size2()}")
                print(f"线性约束向量B: {subproblem.B.size1()}×{subproblem.B.size2()}")

            if subproblem.A.size1() != subproblem.B.size1():
                throw_error(f"子问题{i}的约束矩阵A和B的行数不一致")

            if subproblem.A.size2() != subproblem.get_xs().size1():
                throw_error(f"子问题{i}的约束矩阵A的列数与变量维度不一致")

            lagrange_multiplier_counter += subproblem.A.size1()

        if lagrange_multiplier_counter != self.U_sym.size1():
            throw_error(f"拉格朗日乘子维度({self.U_sym.size1()})与线性约束总数({lagrange_multiplier_counter})不一致")

    def set_rho(self, rho: float):
        """设置ADMM惩罚参数"""
        if rho <= 0:
            raise ValueError("惩罚参数rho必须为正数")
        self._rho = rho

    def set_alpha(self, alpha: float):
        """设置梯度下降步长参数"""
        if alpha <= 0:
            raise ValueError("步长参数alpha必须为正数")
        self._alpha = alpha

    def set_verbose(self, verbose: bool):
        """设置是否输出详细信息"""
        self._verbose = verbose

    def set_adaptive_rho(self, adaptive: bool):
        """设置是否使用自适应惩罚参数"""
        self._adaptive_rho = adaptive

    def get_subproblem(self, index: int) -> OptimizationProblem:
        """获取指定索引的子问题"""
        if index < 0 or index >= len(self._subproblems):
            raise ValueError(f"子问题索引{index}超出范围")
        return self._subproblems[index]

    def get_subproblems(self) -> List[OptimizationProblem]:
        """获取所有子问题"""
        return self._subproblems

    def get_convergence_history(self) -> Dict[str, List[float]]:
        """获取收敛历史"""
        return {
            'primal_residuals': self._primal_residuals,
            'dual_residuals': self._dual_residuals
        }
