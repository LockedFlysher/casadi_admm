import casadi as ca
import numpy as np
from numpy.f2py.auxfuncs import throw_error
from sympy.testing.pytest import warns
import matplotlib.pyplot as plt

# note ： Python中双下划线开头的属性会被自动重命名（名称修饰），导致外部访问失败。
class OptimizationProblemConfiguration:
    def __init__(self, configuration):
        self.variables = configuration['variables']                  # 改为单下划线
        self.objective_function = configuration['objective_function']
        self.equality_constraints = configuration['equality_constraints']
        self.inequality_constraints = configuration['inequality_constraints']
        self.initial_guess = configuration['initial_guess']



# note ： 带有S的都是向量
# 本问题是基于凸优化问题的假设的，采取了对偶上升法的优化求解类
class OptimizationProblem:
    def __init__(self, configuration=None):
        # 变量相关
        self.__alpha__ = 0.5
        self.__augmented_penalty__ = 0.5
        # 变量的初始值和迭代值，后续转换为DM进行运算
        self.__x_k__ = None
        # 符号变量列表
        self.__xs__ = None
        self.__num_of_variables__ = None
        # 不等式约束的lambda，lambdas是用来进行符号计算的，lambda_k是数值计算的
        self.__lambdas__ = None
        self.__lambda_k__ = None
        # 等式约束的mu,mus是用来进行符号计算的，mu_k是数值计算的
        self.__mus__ = None
        self.__mu_k__ = None

        # 目标项累加得到目标方程，有三种方式，一种是直接设置目标函数方程，一种是添加函数项，一种是目标函数+函数项
        self.__objective_expression__ = ca.SX.zeros(1)
        # 约束方程，用来评估对偶上升的情况下对于各条件的违反程度
        self.__equality_constraints__ = []
        self.__inequality_constraints__ = []
        # 获得拉格朗日方程
        self.__lagrange_function__ = ca.SX.zeros(1)
        self.__augmented_lagrange_function__ = ca.SX.zeros(1)

        # 成果
        self.__next_x_function__ = None
        self.__next_multiplier_function__ = None
        self.__objective_expression__ = None
        self.__objective_function__ = None

        self.__objective_expression_set__ = False
        self.__dual_ascent_function_generated__ = False
        self.__multiplier_defined__ = False
        self.__variable_defined__ = False
        self.__initial_guess_set__ = False
        self.__use_augmented_lagrange_function__ = True

        self.__next_mus__ = None
        self.__next_lambdas__ = None
        self.__next_xs__ = None
        self.__next_x_function__ = None
        self.__next_mu_lambda__ = None
        self.__use_configuration__ = False

        if configuration is not None:
            self.set_objective_function(configuration.objective_function)
            self.set_variables(configuration.variables)
            for inequality_constraint in configuration.inequality_constraints:
                self.add_inequality_constraint(inequality_constraint)
            for equality_constraint in configuration.equality_constraints:
                self.add_equality_constraint(equality_constraint)
            self.set_initial_guess(configuration.initial_guess)
            self.__use_configuration__ = True
            self.__objective_expression_set__ = True
            self.__variable_defined__ = True


    def dual_ascent(self, step_num, use_augmented_lagrange_function=False, plot=False):
        """
        对偶上升法求解优化问题

        Args:
            step_num: 迭代步数
            use_augmented_lagrange_function: 是否使用增广拉格朗日函数
            plot: 是否绘制收敛过程

        Returns:
            优化结果字典
        """
        self.__use_augmented_lagrange_function__ = use_augmented_lagrange_function
        self.generate_dual_ascent_function()
        self.set_initial_guess()

        has_equality = len(self.__equality_constraints__) > 0
        has_inequality = len(self.__inequality_constraints__) > 0

        for i in range(step_num):
            # 执行迭代步骤
            if has_equality and has_inequality:
                self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__mu_k__, self.__lambda_k__)
                self.__mu_k__, self.__lambda_k__ = self.__next_multiplier_function__(self.__x_k__, self.__mu_k__,
                                                                                     self.__lambda_k__)
            elif has_equality:
                self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__mu_k__)
                self.__mu_k__ = self.__next_multiplier_function__(self.__x_k__, self.__mu_k__)
            elif has_inequality:
                self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__lambda_k__)
                self.__lambda_k__ = self.__next_multiplier_function__(self.__x_k__, self.__lambda_k__)
            else:
                self.__x_k__ = self.__next_x_function__(self.__x_k__)

        # 返回结果
        result = {'x': self.__x_k__}
        if has_equality:
            result['mu'] = self.__mu_k__
        if has_inequality:
            result['lambda'] = self.__lambda_k__
        return result

    def set_variables(self, variables):
        self.__xs__ = ca.vertcat(*variables)
        self.__num_of_variables__ = len(variables)
        self.__objective_function__ = ca.Function('objective_function',
                                                  [self.__xs__],
                                                  [self.__objective_expression__])
        self.__variable_defined__ = True

    def generate_dual_ascent_function(self):
        if self.__dual_ascent_function_generated__:
            return

        # 1. 初始化乘子
        self.initialize_multipliers()

        # 2. 获取拉格朗日函数
        lagrange_func = self.get_lagrange_function()

        equality_terms = ca.SX.zeros(1)
        inequality_terms = ca.SX.zeros(1)
        if self.__use_augmented_lagrange_function__:
            for h in self.__equality_constraints__:
                equality_terms += (self.__augmented_penalty__ / 2) * h ** 2
            # 添加不等式约束的二次惩罚项
            for g in self.__inequality_constraints__:
                inequality_terms += (self.__augmented_penalty__ / 2) * ca.fmax(0, g) ** 2

        augmented_lagrange_func = lagrange_func + equality_terms + inequality_terms

        # 3. 计算关于x的梯度
        grad_l_x = ca.gradient(augmented_lagrange_func, self.__xs__)
        self.__next_xs__ = self.__xs__ - self.__alpha__ * grad_l_x

        # 4. 处理约束条件
        has_equality = len(self.__equality_constraints__) > 0
        has_inequality = len(self.__inequality_constraints__) > 0

        # 5. 根据约束条件的存在情况创建不同的函数
        if has_equality and has_inequality:
            # 两种约束都存在
            equality_constraints = ca.vertcat(*self.__equality_constraints__)
            inequality_constraints = ca.vertcat(*self.__inequality_constraints__)

            self.__next_mus__ = self.__mus__ + self.__alpha__ * equality_constraints
            self.__next_lambdas__ = ca.fmax(0, self.__lambdas__ + self.__alpha__ * inequality_constraints)

            self.__next_x_function__ = ca.Function('next_x_function',
                                                   [self.__xs__, self.__mus__, self.__lambdas__],
                                                   [self.__next_xs__])
            self.__next_multiplier_function__ = ca.Function('next_multiplier_function',
                                                            [self.__xs__, self.__mus__, self.__lambdas__],
                                                            [self.__next_mus__, self.__next_lambdas__])

        elif has_equality:
            # 只有等式约束
            equality_constraints = ca.vertcat(*self.__equality_constraints__)
            self.__next_mus__ = self.__mus__ + self.__alpha__ * equality_constraints

            self.__next_x_function__ = ca.Function('next_x_function',
                                                   [self.__xs__, self.__mus__],
                                                   [self.__next_xs__])
            self.__next_multiplier_function__ = ca.Function('next_multiplier_function',
                                                            [self.__xs__, self.__mus__],
                                                            [self.__next_mus__])

        elif has_inequality:
            # 只有不等式约束
            inequality_constraints = ca.vertcat(*self.__inequality_constraints__)
            self.__next_lambdas__ = ca.fmax(0, self.__lambdas__ + self.__alpha__ * inequality_constraints)

            self.__next_x_function__ = ca.Function('next_x_function',
                                                   [self.__xs__, self.__lambdas__],
                                                   [self.__next_xs__])
            self.__next_multiplier_function__ = ca.Function('next_multiplier_function',
                                                            [self.__xs__, self.__lambdas__],
                                                            [self.__next_lambdas__])

        else:
            # 没有约束
            self.__next_x_function__ = ca.Function('next_x_function',
                                                   [self.__xs__],
                                                   [self.__next_xs__])
            # 无约束问题不需要更新乘子
            self.__next_multiplier_function__ = None

        self.__dual_ascent_function_generated__ = True

    # 累加所有的目标项，得到目标函数的符号表达式，但是需要一个flag，决定是不是使用了累加
    def get_objective_expression(self):
        # 情况1： 目标函数没有，但是通过加额外项设置目标函数
        if not self.__objective_expression_set__:
            throw_error('未添加目标函数')
        else:
            return self.__objective_expression__

    # 添加等式约束h(x)=0
    def add_equality_constraint(self, equality_constraint):
        self.__equality_constraints__.append(equality_constraint)

    # 添加不等式约束g(x)<=0
    def add_inequality_constraint(self, inequality_constraint):
        self.__inequality_constraints__.append(inequality_constraint)

    # 获取所有等式约束
    def get_equality_constraints(self):
        return self.__equality_constraints__

    # 获取所有不等式约束
    def get_inequality_constraints(self):
        return self.__inequality_constraints__

    # 初始化拉格朗日乘子
    def initialize_multipliers(self):
        if not self.__multiplier_defined__:
            # 初始化等式约束的拉格朗日乘子
            if len(self.__equality_constraints__) > 0:
                if self.__mus__ is None:  # 只有在未初始化时才初始化
                    if len(self.__equality_constraints__) > 1:
                        self.__mus__ = ca.SX.sym('mu_', len(self.__equality_constraints__))
                    else:
                        # 讨论没有等式约束的情况
                        self.__mus__ = ca.SX.sym('mu_0')
            else:
                self.__mus__ = None  # 如果没有等式约束，设置为空列表
            # 初始化不等式约束的拉格朗日乘子
            if len(self.__inequality_constraints__) > 0:
                if self.__lambdas__ is None:  # 只有在未初始化时才初始化
                    if len(self.__inequality_constraints__) > 1:
                        self.__lambdas__ = ca.SX.sym('lambda_', len(self.__inequality_constraints__))
                    else:
                        self.__lambdas__ = ca.SX.sym('lambda_0')
            else:
                self.__lambdas__ = None  # 如果没有不等式约束，设置为空列表
        else:
            warns('请不要多次尝试尝试初始化拉格朗日乘子')

        self.__multiplier_defined__ = True

    def set_objective_function(self, objective_expression):
        self.__objective_expression__ = objective_expression
        self.__objective_expression_set__ = True

    # 构建拉格朗日函数
    def get_lagrange_function(self):
        # 首先确保目标函数已经构建
        obj_func = self.get_objective_expression()
        self.initialize_multipliers()
        # 构建拉格朗日函数
        self.__lagrange_function__ = obj_func
        # 添加等式约束项hx
        for i, h in enumerate(self.__equality_constraints__):
            self.__lagrange_function__ += self.__mus__[i] * h
        # 添加不等式约束项gx
        for i, g in enumerate(self.__inequality_constraints__):
            self.__lagrange_function__ += self.__lambdas__[i] * g
        return self.__lagrange_function__

    # 获取所有拉格朗日乘子
    def get_multipliers(self):
        # 如果乘子还没有初始化，先初始化它们
        self.initialize_multipliers()
        return {
            'equality_multipliers': self.__mus__,
            'inequality_multipliers': self.__lambdas__
        }

    def set_initial_guess(self, initial_guess=None):
        if initial_guess is not None:
            self.__x_k__ = initial_guess
        else:
            if self.__variable_defined__:
                self.__x_k__ = ca.DM.zeros(self.__xs__.size1())
            else:
                throw_error('未设置变量')
            if self.__multiplier_defined__:
                if self.__mus__ is not None:
                    self.__mu_k__ = ca.DM.zeros(self.__mus__.size1())
                if self.__lambdas__ is not None:
                    self.__lambda_k__ = ca.DM.zeros(self.__lambdas__.size1())
            else:
                throw_error('未生成拉格朗日函数')
        self.__initial_guess_set__ = True

    def set_mu_and_lambda(self, mu_, lambda_):
        self.__mu_k__ = mu_
        self.__lambda_k__ = lambda_

    def compute_next_x(self, x, mu_, lambda_):
        # 覆盖一下所有的变量
        self.__x_k__ = x
        self.__mu_k__ = mu_
        self.__lambda_k__ = lambda_

        self.generate_dual_ascent_function()
        self.set_initial_guess()

        has_equality = len(self.__equality_constraints__) > 0
        has_inequality = len(self.__inequality_constraints__) > 0

        # 执行迭代步骤
        if has_equality and has_inequality:
            self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__mu_k__, self.__lambda_k__)
            self.__mu_k__, self.__lambda_k__ = self.__next_multiplier_function__(self.__x_k__, self.__mu_k__,
                                                                                 self.__lambda_k__)
        elif has_equality:
            self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__mu_k__)
            self.__mu_k__ = self.__next_multiplier_function__(self.__x_k__, self.__mu_k__)
        elif has_inequality:
            self.__x_k__ = self.__next_x_function__(self.__x_k__, self.__lambda_k__)
            self.__lambda_k__ = self.__next_multiplier_function__(self.__x_k__, self.__lambda_k__)
        else:
            self.__x_k__ = self.__next_x_function__(self.__x_k__)
        # 返回结果
        result = {'x': self.__x_k__}
        if has_equality:
            result['mu'] = self.__mu_k__
        if has_inequality:
            result['lambda'] = self.__lambda_k__
        return result
