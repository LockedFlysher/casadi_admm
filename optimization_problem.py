import casadi as ca

# note ： 带有S的都是向量
class OptimizationProblem:
    def __init__(self):
        # 目标项，用目标项累加得到目标方程，目前假设的是凸优化问题
        self.__objective_terms_ = []
        self.__objective_function__ = ca.SX.zeros(1)

        # 约束方程
        self.__equality_constraints__ = []
        self.__inequality_constraints__ = []

        # 求解使用到增广的拉格朗日方程
        self.__lagrange_function__ = ca.SX.zeros(1)
        self.__augmented_lagrange_function__ = ca.SX.zeros(1)
        # 不等式约束的lambda
        self.__lambdas__ = None
        # 等式约束的mu
        self.__mus__ = None

        # 罚因子
        self.__rho__ = 1.0

    # 添加目标项到目标函数中
    def add_objective_term(self, term):
        self.__objective_terms_.append(term)

    # 累加所有的目标项，得到目标函数的符号表达式
    def get_objective_function(self):
        self.__objective_function__ = ca.SX.zeros(1)
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
            self.__mus__ = [ca.SX.sym(f'mu_{i}') for i in range(len(self.__equality_constraints__))]

        # 初始化不等式约束的拉格朗日乘子
        if len(self.__inequality_constraints__) > 0:
            self.__lambdas__ = [ca.SX.sym(f'lambda_{i}') for i in range(len(self.__inequality_constraints__))]

    # 构建拉格朗日函数
    def get_lagrange_function(self):
        # 首先确保目标函数已经构建
        obj_func = self.get_objective_function()

        # 初始化乘子（如果还没初始化）
        if self.__mus__ is None or self.__lambdas__ is None:
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

    # 构建增广拉格朗日函数
    def get_augmented_lagrange_function(self):
        # 首先获取普通的拉格朗日函数
        lagrange_func = self.get_lagrange_function()

        # 构建增广项
        augmented_terms = ca.SX.zeros(1)

        # 等式约束的二次惩罚项
        for h in self.__equality_constraints__:
            augmented_terms += (self.__rho__ / 2) * h ** 2

        # 不等式约束的二次惩罚项
        for g in self.__inequality_constraints__:
            augmented_terms += (self.__rho__ / 2) * ca.fmax(0, g) ** 2

        self.__augmented_lagrange_function__ = lagrange_func + augmented_terms
        return self.__augmented_lagrange_function__

    # 设置罚因子
    def set_penalty_parameter(self, rho):
        self.__rho__ = rho

    # 获取罚因子
    def get_penalty_parameter(self):
        return self.__rho__

    # 获取所有拉格朗日乘子
    def get_multipliers(self):
        return {
            'equality_multipliers': self.__mus__,
            'inequality_multipliers': self.__lambdas__
        }
