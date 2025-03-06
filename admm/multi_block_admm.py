from casadi import vertcat
from numpy.f2py.auxfuncs import throw_error
from sympy.testing.pytest import warns

from dual_ascent.dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
from typing import Dict, List, Any


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
        self._rho = 0.15  # 惩罚参数
        self._alpha = 0.05  # 梯度步长参数
        self._adaptive_rho = True  # 是否使用自适应惩罚参数
        self._verbose = True  # 是否输出详细信息

        self._A = None
        # 用来动态地更新残差，用相邻的两项更新残差，减去上一步未更新项的值、加上被更新项的值
        # self._A_list = []
        self._B = None

        self._Xk = None
        # ADMM管理整个问题的约束协调的缩放后的拉格朗日变量
        self._U_sym = None  # 对偶变量，就是y/rho，这个是ADMM管理器2需要更新的参数
        self._Uk = None
        # 更新x的函数，用到上层的U对其进行更新，总共需要的函数是有 【约束的组数 + 1】 个，达成并行的更新
        self._update_x_functions = []  # 更新各子问题x的函数列表
        self._update_u_function = None  # 更新对偶变量的函数
        self._residual = None

        # 子问题列表内可以访问子问题的矩阵A和B，用来计算原问题的残差，更新变量和乘子的时候都要使用到
        self._subproblems = []  # 子问题列表
        self._augmented_lagrange_term = ca.SX.zeros(1)

        # 收敛的评判标准有两个，一个是原问题的残差[Ax-c]^T [Ax-c]、对偶的是对每一组变量求梯度，梯度要接近于0才对
        self._primal_residuals = []
        self._dual_residuals = []
        # 初始化超参数历史记录
        self._rho_history = []
        self._alpha_history = []
        self._subproblem_xs_list = []
        self._linear_constraint_set = False

    def add_subproblem(self, config: OptimizationProblemConfiguration):
        """
        添加子问题，只需要添加：
        1.变量列表
        2.不等式约束
        3.初始的猜测解
        Args:
            config: 子问题的配置
        """
        subproblem = OptimizationProblem(config)
        self._subproblem_xs_list.append(subproblem.get_xs())
        self._subproblems.append(subproblem)

    def set_linear_equality_constraint(self, constraint_A: ca.DM, constraint_B: ca.DM):
        if not self._linear_constraint_set:
            self._A = constraint_A
            # A 是一个矩阵，B 是一个向量
            self._B = constraint_B
            if self._A.size1() != self._B.size1():
                warns("A\B矩阵维度不匹配，约束的数量需要一致")
            xs = ca.vertcat(*self._subproblem_xs_list)
            # 残差项是一个向量
            self._residual = ca.mtimes(self._A, xs) - self._B
            print("残差项已经计算了")
            sub_block_index = 0
            for i, subproblem in enumerate(self._subproblems):
                block_size = subproblem.get_xs().size1()
                self._augmented_lagrange_term += subproblem.get_objective_expression()
                # self._A_list.append(self._A[:,ca.Slice(sub_block_index,sub_block_index+block_size)])
                sub_block_index += block_size

            self._U_sym = ca.SX.sym("U", self._A.size1())
            self._augmented_lagrange_term += ca.sumsqr(self._rho * self._residual + self._U_sym)
            self._linear_constraint_set = True
        else:
            warns("线性约束已经设置完成了不要重复设置")

    def get_augmented_lagrange_function(self):
        return self._augmented_lagrange_term

    def generate_admm_functions(self):
        """
        生成多块ADMM算法所需的函数 - 使用梯度下降方法（保留原有的封闭形式解）
        """
        # 子问题在设计的时候，需要保持子问题的目标函数的加和是和总问题一致的
        # 1.首要目标是求拉格朗日函数的符号表达式、残差的符号表达式，后续的更新都是靠他们

        # todo: 残差项的计算应该是通过一个大的整的A来进行计算的！包括U的计算也是，我们需要先构建起来大的A、B矩阵再计算残差项
        # todo: 残差项的更新采用简单方法更新，不能每一次都计算所有的残差块，省一点时间，需要做一下图

        if not self._linear_constraint_set:
            throw_error("线性约束没有施加")

        # 残差是向量，是和约束的数量一致的，U是协调变量，U的元素的数量是和约束的数量相等的
        self._augmented_lagrange_term = self.get_augmented_lagrange_function()
        # 2.建立子问题的x的梯度的公式，通过一次梯度更新可以得到更新后的x的值，导出公式，放到Function的列表里
        self._update_x_functions = []  # 清空之前的函数
        for i, subproblem in enumerate(self._subproblems):
            gradient = ca.gradient(self._augmented_lagrange_term, subproblem.get_xs())
            next_subproblem_x = subproblem.get_xs() - self._alpha * self._rho * gradient
            # 更新函数
            subproblem_x_update_function = ca.Function(
                f"next_x_function_{i}",
                [self._U_sym, *self._subproblem_xs_list],
                [next_subproblem_x]
            )
            self._update_x_functions.append(subproblem_x_update_function)

            if self._verbose:
                print(f"第 {i} 个x更新方程建立，输入为Uk、当前的Xk，输出为当前子问题的X_i_k+1")

        # 3.建立对偶变量更新函数
        next_u = self._U_sym + self._alpha * ca.mtimes(self._A, ca.vertcat(*self._subproblem_xs_list)) - self._B

        self._update_u_function = ca.Function(
            "next_u_function",
            [self._U_sym, *self._subproblem_xs_list],
            [next_u]
        )

        if self._verbose:
            print("U更新方程建立，输入为Uk、经过更新后的Xk+1，输出为U_{k+1}")

    def solve(self, max_iter: int = 10000, tol: float = 1e-2) -> Dict[str, Any]:
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
        self._Xk = []
        self._Uk = [ca.DM.zeros(self._U_sym.size1())]
        self._primal_residuals = []
        self._dual_residuals = []

        for i, subproblem in enumerate(self._subproblems):
            self._Xk.append(ca.DM(subproblem.get_initial_guess()))

        # 迭代求解
        for iterator in range(max_iter):
            # 记录当前超参数
            self._rho_history.append(self._rho)
            self._alpha_history.append(self._alpha)
            # 保存旧值用于计算残差
            x_old = self._Xk.copy()
            u_old = self._Uk.copy()

            # 更新每个子问题的变量（使用梯度下降）
            for i, subproblem in enumerate(self._subproblems):
                self._Xk[i] = self._update_x_functions[i](*self._Uk, *self._Xk)

            # 更新对偶变量
            self._Uk = [self._update_u_function(*self._Uk, *self._Xk)]

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
                    # 构建结果
                    result = {
                        'x': [x.full() for x in self._Xk],
                        'u': self._Uk,
                        'iterations': iterator + 1,
                        'primal_residual': primal_res,
                        'dual_residual': dual_res,
                        'convergence_history': self.get_convergence_history()
                    }
                    return result

            # 自适应调整参数
            if self._adaptive_rho and iterator > 0 and iterator % 5 == 0:
                self._update_parameters(primal_res, dual_res, iterator)

            # 如果残差变得过大，说明可能发散了
            if primal_res > 1e10 or dual_res > 1e10:
                if self._verbose:
                    print("警告: 残差过大，算法可能发散。尝试减小步长参数alpha和增大惩罚参数rho。")
                self._alpha *= 0.1  # 减小步长
                self._rho *= 1.1  # 增大惩罚参数
                break

        if iterator == max_iter - 1 and self._verbose:
            print(f"ADMM在达到最大迭代次数 {max_iter} 后停止")

        # 构建结果
        result = {
            'x': [x.full() for x in self._Xk],
            'u': self._Uk,
            'iterations': iterator + 1,
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'convergence_history': self.get_convergence_history()
        }

        return result

    def _calculate_primal_residual(self):
        """计算原问题残差"""
        residual = ca.sumsqr(ca.mtimes(self._A, ca.vertcat(*self._Xk)) - self._B)
        return float(residual)

    def _calculate_dual_residual(self, x_old):
        """计算对偶残差"""
        dual_res = self._rho * ca.sumsqr(ca.vertcat(*self._Xk) - ca.vertcat(*x_old))
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
            self._Uk = [self._Uk[0] * (old_rho / self._rho)]
            if self._verbose:
                print(f"调整惩罚参数: rho从 {old_rho} 更新到 {self._rho}")

        # 动态调整步长alpha（基于残差）
        old_alpha = self._alpha
        if iteration > 20:  # 前几次迭代不调整
            if len(self._primal_residuals) >= 3:
                # 如果残差持续增加，减小步长
                if (self._primal_residuals[-1] > self._primal_residuals[-2] and
                        self._primal_residuals[-2] > self._primal_residuals[-3]):
                    self._alpha *= 0.9
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
