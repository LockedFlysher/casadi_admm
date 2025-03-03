from audioop import error

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple

from sympy.testing.pytest import warns


class OptimizationProblemConfiguration:
    """优化问题配置类，用于存储优化问题的各种配置参数"""

    def __init__(self, configuration: Dict[str, Any]):
        self.variables = configuration['variables']
        self.objective_function = configuration['objective_function']
        self.equality_constraints = configuration['equality_constraints']
        self.inequality_constraints = configuration['inequality_constraints']
        self.initial_guess = configuration['initial_guess']


class OptimizationProblem:
    """
    基于凸优化问题假设的优化求解类，采用对偶上升法
    """

    def __init__(self, configuration: Optional[OptimizationProblemConfiguration] = None):
        """
        初始化优化问题求解器

        Args:
            configuration: 可选的优化问题配置
        """
        # 算法参数
        self._alpha = 0.05  # 步长
        self._augmented_equality_penalty = 2  # 增广拉格朗日惩罚因子
        self._augmented_inequality_penalty = 50
        self.A = None
        self.C = None

        # 变量相关
        self._x_k = None  # 变量的当前值
        self._xs = None  # 符号变量
        self._num_of_variables = None  # 变量数量

        # 拉格朗日乘子相关
        self._mus = None  # 等式约束的符号乘子
        self._mu_k = None  # 等式约束的数值乘子

        # 目标函数和约束
        self._objective_expression = ca.SX.zeros(1)  # 目标函数表达式
        self._equality_constraints = []  # 等式约束列表
        self._inequality_constraints = []  # 不等式约束列表

        # 拉格朗日函数
        self._lagrange_function = ca.SX.zeros(1)  # 拉格朗日函数
        self._augmented_lagrange_function = ca.SX.zeros(1)  # 增广拉格朗日函数

        # 迭代函数
        self._next_x_function = None  # 下一步 x 的计算函数
        self._next_multiplier_function = None  # 下一步乘子的计算函数
        self._objective_function = None  # 目标函数

        # 状态标志
        self._objective_expression_set = False  # 目标函数是否已设置
        self._dual_ascent_function_generated = False  # 对偶上升函数是否已生成
        self._multiplier_defined = False  # 乘子是否已定义
        self._variable_defined = False  # 变量是否已定义
        self._initial_guess_set = False  # 初始猜测是否已设置
        self._use_augmented_lagrange_function = True  # 是否使用增广拉格朗日函数

        # 中间变量
        self._next_mus = None  # 下一步 mu 的表达式
        self._next_lambdas = None  # 下一步 lambda 的表达式
        self._next_xs = None  # 下一步 x 的表达式
        self._next_mu_lambda = None  # 下一步 mu 和 lambda 的表达式
        self._use_configuration = False  # 是否使用配置

        # 如果提供了配置，则使用配置初始化
        if configuration is not None:
            self.set_objective_function(configuration.objective_function)
            self.set_variables(configuration.variables)
            for inequality_constraint in configuration.inequality_constraints:
                self.add_inequality_constraint(inequality_constraint)
            if len(configuration.equality_constraints["A"]) == len(configuration.equality_constraints["C"]):
                self.A = ca.vertcat(*configuration.equality_constraints["A"]).T
                self.C = ca.vertcat(*configuration.equality_constraints["C"])
                for A_line,C_term in zip(configuration.equality_constraints["A"],configuration.equality_constraints["C"]):
                    self.add_equality_constraint(ca.mtimes(A_line.T,self._xs)- C_term)
            else:
                error("等式矩阵的维度不对")
            self.set_initial_guess(configuration.initial_guess)
            self._use_configuration = True
            self._objective_expression_set = True
            self._variable_defined = True

    def dual_ascent(self, step_num: int, use_augmented_lagrange_function: bool = False,
                    plot: bool = False) -> Dict[str, ca.DM]:
        """
        对偶上升法求解优化问题

        Args:
            step_num: 迭代步数
            use_augmented_lagrange_function: 是否使用增广拉格朗日函数
            plot: 是否绘制收敛过程

        Returns:
            优化结果字典，包含优化变量和拉格朗日乘子
        """
        self._use_augmented_lagrange_function = use_augmented_lagrange_function
        self.generate_dual_ascent_function()
        self.set_initial_guess()

        has_equality = len(self._equality_constraints) > 0

        for i in range(step_num):
            # 执行迭代步骤
            if has_equality:
                self._x_k = self._next_x_function(self._x_k, self._mu_k)
                self._mu_k = self._next_multiplier_function(self._x_k, self._mu_k)
            else:
                self._x_k = self._next_x_function(self._x_k)

        # 返回结果
        result = {'x': self._x_k}
        if has_equality:
            result['mu'] = self._mu_k
        return result

    def set_variables(self, variables: List[ca.SX]):
        """
        设置优化变量

        Args:
            variables: 优化变量列表
        """
        self._xs = ca.vertcat(*variables)
        self._num_of_variables = len(variables)
        self._objective_function = ca.Function('objective_function',
                                               [self._xs],
                                               [self._objective_expression])
        self._variable_defined = True

    def generate_dual_ascent_function(self):
        """
        生成对偶上升函数
        """
        if self._dual_ascent_function_generated:
            return

        # 1. 初始化乘子
        self.initialize_multipliers()

        # 2. 获取拉格朗日函数
        lagrange_func = self.get_lagrange_function()

        equality_terms = ca.SX.zeros(1)
        # 添加不等式约束的二次惩罚项

        if self._use_augmented_lagrange_function:
            for h in self._equality_constraints:
                equality_terms += (self._augmented_equality_penalty / 2) * h ** 2

        augmented_lagrange_func = lagrange_func + equality_terms

        # 3. 计算关于x的梯度
        grad_l_x = ca.gradient(augmented_lagrange_func, self._xs)
        self._next_xs = self._xs - self._alpha * grad_l_x

        # 4. 处理约束条件
        has_equality = len(self._equality_constraints) > 0

        # 5. 根据约束条件的存在情况创建不同的函数
        if has_equality:
            # 只有等式约束
            equality_constraints = ca.vertcat(*self._equality_constraints)
            self._next_mus = self._mus + self._alpha * equality_constraints

            self._next_x_function = ca.Function('next_x_function',
                                                [self._xs, self._mus],
                                                [self._next_xs])
            self._next_multiplier_function = ca.Function('next_multiplier_function',
                                                         [self._xs, self._mus],
                                                         [self._next_mus])

        else:
            # 没有约束
            self._next_x_function = ca.Function('next_x_function',
                                                [self._xs],
                                                [self._next_xs])
            # 无约束问题不需要更新乘子
            self._next_multiplier_function = None

        self._dual_ascent_function_generated = True

    def get_objective_expression(self) -> ca.SX:
        """
        获取目标函数表达式

        Returns:
            目标函数的符号表达式

        Raises:
            ValueError: 如果未添加目标函数
        """
        if not self._objective_expression_set:
            raise ValueError('未添加目标函数')
        return self._objective_expression

    def add_equality_constraint(self, equality_constraint: ca.SX):
        """
        添加等式约束 h(x) = 0

        Args:
            equality_constraint: 等式约束表达式
        """
        self._equality_constraints.append(equality_constraint)

    def get_xs(self):
        return self._xs

    def add_inequality_constraint(self, inequality_constraint: ca.SX):
        """
        添加不等式约束 g(x) <= 0， 作用到拉格朗日函数上，用惩罚代替不等式约束，提高求解的效率

        Args:
            inequality_constraint: 不等式约束表达式
        """
        self._objective_expression += (self._augmented_inequality_penalty / 2) * ca.fmax(0, inequality_constraint) ** 2
        self._inequality_constraints.append(inequality_constraint)

    def get_equality_constraints(self) -> List[ca.SX]:
        """
        获取所有等式约束

        Returns:
            等式约束列表
        """
        return self._equality_constraints

    def get_inequality_constraints(self) -> List[ca.SX]:
        """
        获取所有不等式约束

        Returns:
            不等式约束列表
        """
        return self._inequality_constraints

    def initialize_multipliers(self):
        """
        初始化拉格朗日乘子
        """
        if not self._multiplier_defined:
            # 初始化等式约束的拉格朗日乘子
            if len(self._equality_constraints) > 0:
                if self._mus is None:  # 只有在未初始化时才初始化
                    if len(self._equality_constraints) > 1:
                        self._mus = ca.SX.sym('mu_', len(self._equality_constraints))
                    else:
                        self._mus = ca.SX.sym('mu_0')
            else:
                self._mus = None  # 如果没有等式约束，设置为None
        else:
            import warnings
            warnings.warn('请不要多次尝试初始化拉格朗日乘子')

        self._multiplier_defined = True

    def set_objective_function(self, objective_expression: ca.SX):
        """
        设置优化问题的目标函数

        Args:
            objective_expression: 目标函数表达式
        """
        self._objective_expression = objective_expression
        self._objective_expression_set = True

    def get_lagrange_function(self) -> ca.SX:
        """
        构建拉格朗日函数

        Returns:
            拉格朗日函数表达式
        """
        # 首先确保目标函数已经构建
        obj_func = self.get_objective_expression()
        self.initialize_multipliers()

        # 构建拉格朗日函数
        self._lagrange_function = obj_func

        # 添加等式约束项 h(x)
        if self._mus is not None:
            for i, h in enumerate(self._equality_constraints):
                self._lagrange_function += self._mus[i] * h

        return self._lagrange_function

    def get_multipliers(self) -> Dict[str, Union[ca.SX, None]]:
        """
        获取所有拉格朗日乘子

        Returns:
            包含等式和不等式约束乘子的字典
        """
        # 如果乘子还没有初始化，先初始化它们
        self.initialize_multipliers()
        return {
            'equality_multipliers': self._mus,
            'inequality_multipliers': self._lambdas
        }

    def set_initial_guess(self, initial_guess: Optional[ca.DM] = None):
        """
        设置优化变量和乘子的初始猜测值

        Args:
            initial_guess: 优化变量的初始值，如果为None则使用零向量

        Raises:
            ValueError: 如果未设置变量或未生成拉格朗日函数
        """
        if initial_guess is not None:
            self._x_k = initial_guess
        else:
            if self._variable_defined:
                self._x_k = ca.DM.zeros(self._xs.size1())
            else:
                raise ValueError('未设置变量')

            if self._multiplier_defined:
                if self._mus is not None:
                    self._mu_k = ca.DM.zeros(self._mus.size1())
            else:
                raise ValueError('未生成拉格朗日函数')

        self._initial_guess_set = True

    def set_mu_and_lambda(self, mu_: ca.DM, lambda_: ca.DM):
        """
        设置拉格朗日乘子的值

        Args:
            mu_: 等式约束乘子的值
            lambda_: 不等式约束乘子的值
        """
        self._mu_k = mu_
        self._lambda_k = lambda_

    def compute_next_x(self, x: ca.DM, mu_: ca.DM, lambda_: ca.DM) -> Dict[str, ca.DM]:
        """
        计算下一步的优化变量和乘子值

        Args:
            x: 当前优化变量值
            mu_: 当前等式约束乘子值
            lambda_: 当前不等式约束乘子值

        Returns:
            包含更新后的优化变量和乘子的字典
        """
        # 更新当前值
        self._x_k = x
        self._mu_k = mu_
        self._lambda_k = lambda_

        # 确保已生成对偶上升函数
        self.generate_dual_ascent_function()
        self.set_initial_guess()

        has_equality = len(self._equality_constraints) > 0
        has_inequality = len(self._inequality_constraints) > 0

        # 执行迭代步骤
        if has_equality and has_inequality:
            self._x_k = self._next_x_function(self._x_k, self._mu_k, self._lambda_k)
            self._mu_k, self._lambda_k = self._next_multiplier_function(self._x_k, self._mu_k, self._lambda_k)
        elif has_equality:
            self._x_k = self._next_x_function(self._x_k, self._mu_k)
            self._mu_k = self._next_multiplier_function(self._x_k, self._mu_k)
        elif has_inequality:
            self._x_k = self._next_x_function(self._x_k, self._lambda_k)
            self._lambda_k = self._next_multiplier_function(self._x_k, self._lambda_k)
        else:
            self._x_k = self._next_x_function(self._x_k)

        # 返回结果
        result = {'x': self._x_k}
        if has_equality:
            result['mu'] = self._mu_k
        if has_inequality:
            result['lambda'] = self._lambda_k
        return result

    @property
    def xs(self):
        return self._xs
