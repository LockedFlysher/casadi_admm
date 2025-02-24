import casadi as ca
import matplotlib.pyplot as plt
from fsspec.asyn import private
from numpy.array_api import zeros
from numpy.f2py.auxfuncs import throw_error
from sympy.testing.pytest import warns

# note ： 带有S的都是向量
# 本问题是基于凸优化问题的假设的
class OptimizationProblem:
    def __init__(self):
        # 变量相关
        self.__alpha__ = 0.02
        # 变量的初始值和迭代值，后续转换为DM进行运算
        self.__x_k__ = None
        # 符号变量列表
        self.__xs__ = None
        # 不等式约束的lambda，lambdas是用来进行符号计算的，lambda_k是数值计算的
        self.__lambdas__ = None
        self.__lambda_k__ = None
        # 等式约束的mu,mus是用来进行符号计算的，mu_k是数值计算的
        self.__mus__ = None
        self.__mu_k__ = None

        # 目标项累加得到目标方程，有三种方式，一种是直接设置目标函数方程，一种是添加函数项，一种是目标函数+函数项
        self.__objective_terms_ = []
        self.__objective_function__ = ca.SX.zeros(1)
        # 约束方程，用来评估对偶上升的情况下对于各条件的违反程度
        self.__equality_constraints__ = []
        self.__inequality_constraints__ = []
        # 获得拉格朗日方程
        self.__lagrange_function__ = ca.SX.zeros(1)
        # todo 完成增广形式迭代求解
        self.__augmented_lagrange_function__ = ca.SX.zeros(1)

        # 成果
        self.__next_x_function__ = None
        self.__next_multiplier_function__ = None

        self.__objective_function_set__ = False
        self.__dual_ascent_function_generated__ = False
        self.__multiplier_defined__ = False
        self.__variable_defined__ = False
        self.__initial_guess_set__ = False

        self.__next_mus__ = None
        self.__next_lambdas__ = None
        self.__next_xs__ = None
        self.__next_x_function__ = None
        self.__next_mu_lambda__ = None


    def dual_ascent(self,step_num):
        if not self.__dual_ascent_function_generated__:
            self.generate_dual_ascent_function()

        if not self.__initial_guess_set__:
            self.set_initial_guess()

        for i in range(step_num):
            self.__x_k__ = self.__next_x_function__(self.__x_k__,self.__mu_k__,self.__lambda_k__)
            self.__mu_k__,self.__lambda_k__ = self.__next_multiplier_function__(self.__x_k__,self.__mu_k__,self.__lambda_k__)

        return {
            'x': self.__x_k__,
            'mu': self.__mu_k__,
            'lambda': self.__lambda_k__,
        }

    def define_variables(self, *variables):
        self.__xs__ = ca.vertcat(*variables)
        self.__variable_defined__ = True

    def generate_dual_ascent_function(self):
        # 问题1 ： 如何确定刚开始的lambda和mu，全部假设为0吗？---随机初始化，先假设为0
        # 问题2 : x怎么寻找下一个点？ x_{k+1} arg min L(x,y_{k})
        # -- 假设原问题是一个凸的问题，加上的后续的项其实都是线性的项
        # -- 所以合成的问题是凸问题，用梯度求，让此点的梯度为0就行了
        # -- 1. 拉格朗日函数要拿到
        # -- 2. 从拉格朗日函数代入mu和lambda，得到新的表达式
        # -- 3. 新的表达式对x进行求导，结果应该为0向量，得到x的更新量
        # -- 4. 用x的更新量去求y
        # 检查初始化，如果已经生成则不再生成
        if self.__dual_ascent_function_generated__:
            pass
        else:
            # 定义出乘子的表达式
            if not self.__multiplier_defined__:
                self.initialize_multipliers()

            # 获取拉格朗日函数
            lagrange_func = self.get_lagrange_function()

            # 对拉格朗日函数关于x求梯度
            grad_l_x = ca.gradient(lagrange_func, self.__xs__)

            # 创建一个函数，把当前的mu和lambda输入进去，得到x的梯度
            # note: *是解包操作，把列表拆出来成为单独的变量
            print([self.__xs__, self.__mus__, self.__lambdas__])
            self.__next_xs__ = self.__xs__ - self.__alpha__ * grad_l_x

            # 把等式约束、不等式约束放进来
            # 修正：添加空约束的处理
            equality_constraints = ca.vertcat(
                *self.get_equality_constraints()) if self.get_equality_constraints() else ca.SX.zeros(0)
            inequality_constraints = ca.vertcat(
                *self.get_inequality_constraints()) if self.get_inequality_constraints() else ca.SX.zeros(0)

            # 统一向量化处理 (需要确保乘子已初始化为CasADi向量)
            self.__next_mus__ = self.__mus__ + self.__alpha__ * equality_constraints if self.__mus__.size1() > 0 else []
            self.__next_lambdas__ = ca.fmax(0, self.__lambdas__ + self.__alpha__ * inequality_constraints) if self.__lambdas__.size1() > 0 else []

            self.__next_x_function__ = ca.Function('next_x_function',
                                                   [self.__xs__, self.__mus__, self.__lambdas__],
                                                   [self.__next_xs__])
            self.__next_multiplier_function__ = ca.Function('next_multiplier_function',
                                                            [self.__xs__, self.__mus__, self.__lambdas__],
                                                            [self.__next_mus__, self.__next_lambdas__])
            self.__dual_ascent_function_generated__ = True

    # 添加目标项到目标函数中
    def add_objective_term(self, term):
        self.__objective_function_set__ = True
        self.__objective_terms_.append(term)

    # 累加所有的目标项，得到目标函数的符号表达式，但是需要一个flag，决定是不是使用了累加
    def get_objective_function(self):
        for term in self.__objective_terms_:
            self.__objective_function__ += term
        return self.__objective_function__

    # 添加等式约束
    def add_equality_constraint(self, equality_constraint):
        self.__equality_constraints__.append(equality_constraint)

    # 添加不等式约束
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
        # 初始化等式约束的拉格朗日乘子
        if len(self.__equality_constraints__) > 0:
            if self.__mus__ is None:  # 只有在未初始化时才初始化
                self.__mus__ = ca.SX.sym('mu_',len(self.__equality_constraints__))
        else:
            self.__mus__ = None  # 如果没有等式约束，设置为空列表
        # 初始化不等式约束的拉格朗日乘子
        if len(self.__inequality_constraints__) > 0:
            if self.__lambdas__ is None:  # 只有在未初始化时才初始化
                self.__lambdas__ = ca.SX.sym('lambda_',len(self.__inequality_constraints__))
        else:
            self.__lambdas__ = None  # 如果没有不等式约束，设置为空列表

        self.__multiplier_defined__ = True

    def set_objective_function(self, objective_function):
        self.__objective_function__ = objective_function

    # 构建拉格朗日函数
    def get_lagrange_function(self):
        # 首先确保目标函数已经构建
        if not self.__objective_function_set__:
            throw_error('目标函数未设置')
        obj_func = self.get_objective_function()

        # 初始化乘子（如果还没初始化）

        self.initialize_multipliers()

        # 构建拉格朗日函数
        self.__lagrange_function__ = obj_func

        # 添加等式约束项
        for i, h in enumerate(self.__equality_constraints__):
            self.__lagrange_function__ += self.__mus__[i] * h

        # 添加不等式约束项
        for i, g in enumerate(self.__inequality_constraints__):
            self.__lagrange_function__ += self.__lambdas__[i] * g

        return self.__lagrange_function__

    # # 构建增广拉格朗日函数
    # def get_augmented_lagrange_function(self):
    #     # 首先获取普通的拉格朗日函数
    #     lagrange_func = self.get_lagrange_function()
    #
    #     # 构建增广项
    #     augmented_terms = ca.SX.zeros(1)
    #
    #     # 等式约束的二次惩罚项
    #     for h in self.__equality_constraints__:
    #         augmented_terms += (self.__rho__ / 2) * h ** 2
    #
    #     # 不等式约束的二次惩罚项
    #     for g in self.__inequality_constraints__:
    #         augmented_terms += (self.__rho__ / 2) * ca.fmax(0, g) ** 2
    #
    #     self.__augmented_lagrange_function__ = lagrange_func + augmented_terms
    #     return self.__augmented_lagrange_function__

    # 获取所有拉格朗日乘子
    def get_multipliers(self):
        # 如果乘子还没有初始化，先初始化它们
        if not self.__multiplier_defined__:
            self.initialize_multipliers()
            return {
                'equality_multipliers': self.__mus__,
                'inequality_multipliers': self.__lambdas__
            }
        # 返回乘子
        else:
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
                self.__mu_k__ = ca.DM.zeros(self.__mus__.size1())
                self.__lambda_k__ = ca.DM.zeros(self.__lambdas__.size1())
            else:
                throw_error('未生成拉格朗日函数')
        self.__initial_guess_set__ = True


