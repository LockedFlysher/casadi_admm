from casadi import vertcat
from numpy.f2py.auxfuncs import throw_error
from sympy.physics.vector import gradient
from sympy.testing.pytest import warns

from dual_ascent import OptimizationProblem, OptimizationProblemConfiguration
import casadi as ca
from typing import Dict, List, Union, Optional, Any, Tuple

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
        self._rho = 1.0  # 惩罚参数
        self._alpha = 0.1

        # ADMM管理整个问题的约束协调的缩放后的拉格朗日变量
        self.U = None  # 对偶变量，就是y/rho，这个是ADMM管理器2需要更新的参数
        # 更新x的函数，用到上层的U对其进行更新，总共需要的函数是有 【约束的组数 + 1】 个，达成并行的更新
        self._update_x_functions = []  # 更新各子问题x的函数列表
        self._update_u_function = None  # 更新对偶变量的函数

        # 子问题列表内可以访问子问题的矩阵A和B，用来计算原问题的残差，更新变量和乘子的时候都要使用到
        self._subproblems = []  # 子问题列表

        # 收敛的评判标准有两个，一个是原问题的残差[Ax-c]^T [Ax-c]、对偶的是对每一组变量求梯度，梯度要接近于0才对
        self._primal_residuals = []
        self._dual_residuals = []

    def add_subproblem(self, config: OptimizationProblemConfiguration):
        """
        添加子问题

        Args:
            config: 子问题的配置
            A: 约束矩阵A_i，如果为None则默认为单位矩阵
        """
        subproblem = OptimizationProblem(config)
        self._subproblems.append(subproblem)

    def generate_admm_functions(self):
        """
        生成多块ADMM算法所需的函数
        """
        # 1.首要目标是求拉格朗日函数的符号表达式、残差的符号表达式，后续的更新都是靠他们
        augmented_lagrange_function = ca.SX.zeros(1)
        residual_vector = []
        for i,subproblem in enumerate(self._subproblems):
            # 原始的目标函数的求和
            augmented_lagrange_function += subproblem.get_objective_expression()
            # 残差项的计算
            residual_vector.append(subproblem.A @ subproblem.get_xs() - subproblem.B)
        residual = ca.vertcat(*residual_vector)
        augmented_lagrange_function += self._rho/2 *(residual + self.U).T * (residual + self.U)
        # 常量是A\B\C\rho，变量是U、X_i，现在已经求到了拉格朗日函数的缩放后的形式augmented_lagrange_function
        # 2.建立子问题的x的梯度的公式，通过一次梯度更新可以得到更新后的x的值，导出公式，放到Function的列表里
        # 已知： 所有的变量的当前值，已知，求在此点的子问题的变量的梯度，更新的是子问题被分离的变量中的一组向量
        for i,subproblem in enumerate(self._subproblems):
            gradient = ca.gradient(augmented_lagrange_function, subproblem.get_xs())
            next_subproblem_x = subproblem.get_xs() - gradient*self._alpha
            subproblem_x_update_function = ca.Function("next_x_function",
                                                       [self.U,subproblem.get_xs()],
                                                       [next_subproblem_x])
            print("第",i,"个x更新方程建立成功")
            print(subproblem_x_update_function)
            self._update_x_functions.append(subproblem_x_update_function)

        next_whole_problem_u = self.U+self._alpha*residual
        # U的更新依赖于所有的X，对于一个分布式的优化问题，子问题里肯定会出现“子问题内变量的数量加起来比总问题多的问题”
        # 子问题的变量的数量确实一般情况下就是大于总问题的变量数，子问题在设计的时候，需要保持子问题的目标函数的加和是和总问题一致的
        self._update_u_function = ca.Function("next_u_function",
                                              [self.U,self._xs],
                                              [next_whole_problem_u])

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> Dict[str, Any]:
        """
        使用多块ADMM算法求解问题

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            求解结果字典
        """

        # result = {
        #     'x': x_values,
        #     'u': u,
        #     'iterations': k + 1,
        #     'primal_residual': float(primal_res),
        #     'dual_residual': float(dual_res)
        # }
        pass
        # return result

    def check(self):
        # 1.检查子问题的维度，检查Function的构建情况，
        print("检查项1 ： 拉格朗日乘子维度： ")
        lagrange_multiplier_counter = 0
        print(self.U.size1)
        for i, subproblem in enumerate(self._subproblems):
            print("正在检查编号为【" + str(i) + "】的子问题")
            # 检查维度是否和函数和拉格朗日匹配，A矩阵在构建的时候是已经转置过了的
            print("线性约束矩阵A、B")
            print(subproblem.A.size1())
            print(subproblem.A.size2())
            print(subproblem.B.size1())
            print(subproblem.B.size2())
            if subproblem.A.size1() != subproblem.B.size1():
                throw_error("怎么搞的，A、B矩阵的维度都不一致")
            if subproblem.A.size2() != subproblem.get_xs.size1():
                throw_error("怎么搞的，A矩阵的列数和变量的维度不一样")
            lagrange_multiplier_counter += subproblem.A.size1()

        if lagrange_multiplier_counter != self.U.size1():
            throw_error("拉格朗日的维度不能和线性约束的矩阵维度一致")

        pass

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
