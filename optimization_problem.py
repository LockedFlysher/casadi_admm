import casadi as ca
import matplotlib.pyplot as plt
from fsspec.asyn import private


# note ： 带有S的都是向量
# 本问题是基于凸优化问题的假设的
class OptimizationProblem:
    def __init__(self):
        # 目标项，用目标项累加得到目标方程
        self.__objective_terms_ = []
        self.__objective_function__ = ca.SX.zeros(1)

        # 约束方程，用来评估对偶上升的情况下对于各条件的违反程度
        self.__equality_constraints__ = []
        self.__inequality_constraints__ = []

        # 求解使用到增广的拉格朗日方程
        self.__x_k__ = []
        self.__lagrange_function__ = ca.SX.zeros(1)
        # todo 完成增广形式迭代求解
        self.__augmented_lagrange_function__ = ca.SX.zeros(1)
        # 不等式约束的lambda，lambdas是用来进行符号计算的，lambda_k是数值计算的
        self.__lambdas__ = None
        self.__lambda_k__ = None
        # 等式约束的mu,mus是用来进行符号计算的，mu_k是数值计算的
        self.__mus__ = None
        self.__mu_k__ = None

    # 添加目标项到目标函数中
    def add_objective_term(self, term):
        self.__objective_terms_.append(term)

    # 累加所有的目标项，得到目标函数的符号表达式，但是需要一个flag，决定是不是使用了累加
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
            if self.__mus__ is None:  # 只有在未初始化时才初始化
                self.__mus__ = [ca.SX.sym(f'mu_{i}') for i in range(len(self.__equality_constraints__))]
        else:
            self.__mus__ = []  # 如果没有等式约束，设置为空列表
        # 初始化不等式约束的拉格朗日乘子
        if len(self.__inequality_constraints__) > 0:
            if self.__lambdas__ is None:  # 只有在未初始化时才初始化
                self.__lambdas__ = [ca.SX.sym(f'lambda_{i}') for i in range(len(self.__inequality_constraints__))]
        else:
            self.__lambdas__ = []  # 如果没有不等式约束，设置为空列表
        print(self.__mus__,self.__lambdas__)

    # 构建拉格朗日函数
    def get_lagrange_function(self):
        # 首先确保目标函数已经构建
        obj_func = self.get_objective_function()

        # 初始化乘子（如果还没初始化）
        if self.__mus__ is None or self.__lambdas__ is None:
            print('拉格朗日乘子未初始化')
            self.initialize_multipliers()
            print('拉格朗日乘子初始化完成')

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
        # 如果乘子还没有初始化，先初始化它们
        if self.__mus__ is None or self.__lambdas__ is None:
            self.initialize_multipliers()
        # 返回乘子
        return {
            'equality_multipliers': self.__mus__,
            'inequality_multipliers': self.__lambdas__
        }

    def set_initial_guess(self,initial_guess):
        self.__x_k__=initial_guess

    def dual_single_step_ascent(self):
        # 问题1 ： 如何确定刚开始的lambda和mu，全部假设为0吗？---随机初始化，先假设为0
        # 问题2 : x怎么寻找下一个点？ x_{k+1} arg min L(x,y_{k})
        # -- 假设原问题是一个凸的问题，加上的后续的项其实都是线性的项
        # -- 所以合成的问题是凸问题，用梯度求，让此点的梯度为0就行了
        # -- 1. 拉格朗日函数要拿到
        # -- 2. 从拉格朗日函数代入mu和lambda，得到新的表达式
        # -- 3. 新的表达式对x进行求导，结果应该为0向量，得到x的更新量
        # -- 4. 用x的更新量去求y

        # 检查初始化
        if self.__x_k__ is None:
            raise ValueError("Initial guess not set. Call set_initial_guess first.")
        if self.__mus__ is None or self.__lambdas__ is None:
            self.initialize_multipliers()

        # 获取拉格朗日函数
        lagrange_func = self.get_lagrange_function()
        all_vars = ca.symvar(lagrange_func)

        # 提取原始变量（非对偶变量）
        x_vars = []
        for var in all_vars:
            var_name = var.name()
            if not (var_name.startswith('lambda_')) and not (var_name.startswith('mu_')):
                x_vars.append(var)
                print('added x var', var_name)

        if not x_vars:
            raise ValueError("No primal variables found in the Lagrangian")

        x_vec = ca.vertcat(*x_vars)

        # 对拉格朗日函数关于x求梯度
        grad_L_x = ca.gradient(lagrange_func, x_vec)

        # 创建一个函数，把当前的mu和lambda输入进去，得到x的梯度
        # 修正：添加x_vars作为输入
        solve_primal = ca.Function('solve_primal',
                                   [*x_vars, *self.__mus__, *self.__lambdas__],
                                   [grad_L_x])

        # 把等式约束、不等式约束放进来
        # 修正：添加空约束的处理
        equality_constraints = ca.vertcat(
            *self.get_equality_constraints()) if self.get_equality_constraints() else ca.SX.zeros(0)
        inequality_constraints = ca.vertcat(
            *self.get_inequality_constraints()) if self.get_inequality_constraints() else ca.SX.zeros(0)

        # 用来更新对偶变量
        constrain_func = ca.Function('constrains',
                                     [x_vec],
                                     [equality_constraints, inequality_constraints])

        # 步长参数
        alpha = 0.01  # 可以考虑使用自适应步长或作为类的参数

        if (self.__x_k__ == []):
            self.__x_k__ = [0,0]
        if (self.__mu_k__ == None):
            self.__mu_k__ = [0]
        if(self.__lambda_k__ == None):
            self.__lambda_k__ = [0]
        # 求解原问题
        current_x = self.__x_k__
        # 修正：添加current_x作为输入
        grad_val = solve_primal(*current_x, *self.__mu_k__, *self.__lambda_k__)

        # 进行一次梯度下降，更新x_k
        self.__x_k__ = current_x - alpha * grad_val

        # 计算约束违反程度
        h_val, g_val = constrain_func(self.__x_k__)

        # 更新等式约束的乘子
        if len(self.__mu_k__) > 0:
            self.__mu_k__ = [mu + alpha * float(h) for mu, h in zip(self.__mu_k__, h_val.elements())]

        # 更新不等式约束的乘子（保持非负）
        if len(self.__lambda_k__) > 0:
            self.__lambda_k__ = [ca.fmax(0.0, lambda_ + alpha * float(g))
                                 for lambda_, g in zip(self.__lambda_k__, g_val.elements())]

        # 返回当前迭代的结果
        return {
            'primal_variables': self.__x_k__,
            'dual_variables': {
                'mu': self.__mu_k__,
                'lambda': self.__lambda_k__
            },
            'constraints_violation': {
                'equality': h_val,
                'inequality': g_val
            }
        }

    def dual_ascent(self, number_of_steps):
        # 存储优化过程中的历史数据
        history = {
            'primal_vars': [],
            'dual_vars_mu': [],
            'dual_vars_lambda': [],
            'equality_violations': [],
            'inequality_violations': [],
            'iterations': []
        }

        # 执行优化迭代
        for i in range(number_of_steps):
            result = self.dual_single_step_ascent()
            # 记录历史数据
            history['primal_vars'].append(float(result['primal_variables']))
            history['dual_vars_mu'].append([float(mu) for mu in result['dual_variables']['mu']])
            history['dual_vars_lambda'].append([float(lam) for lam in result['dual_variables']['lambda']])
            history['equality_violations'].append([float(h) for h in result['constraints_violation']['equality']])
            history['inequality_violations'].append([float(g) for g in result['constraints_violation']['inequality']])
            history['iterations'].append(i)

        # 创建可视化图表
        plt.figure(figsize=(15, 10))

        # 1. 原变量收敛图
        plt.subplot(2, 2, 1)
        plt.plot(history['iterations'], history['primal_vars'], 'b-', label='Primal Variable')
        plt.title('Primal Variable Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

        # 2. 对偶变量(mu)收敛图
        plt.subplot(2, 2, 2)
        for i, mu_series in enumerate(zip(*history['dual_vars_mu'])):
            plt.plot(history['iterations'], mu_series, label=f'μ_{i}')
        plt.title('Dual Variables (μ) Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

        # 3. 对偶变量(lambda)收敛图
        plt.subplot(2, 2, 3)
        for i, lambda_series in enumerate(zip(*history['dual_vars_lambda'])):
            plt.plot(history['iterations'], lambda_series, label=f'λ_{i}')
        plt.title('Dual Variables (λ) Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

        # 4. 约束违反度
        plt.subplot(2, 2, 4)
        # 等式约束违反度
        for i, eq_series in enumerate(zip(*history['equality_violations'])):
            plt.plot(history['iterations'], eq_series, '--', label=f'Equality_{i}')
        # 不等式约束违反度
        for i, ineq_series in enumerate(zip(*history['inequality_violations'])):
            plt.plot(history['iterations'], ineq_series, ':', label=f'Inequality_{i}')
        plt.title('Constraints Violation')
        plt.xlabel('Iteration')
        plt.ylabel('Violation')
        plt.yscale('log')  # 使用对数坐标以便观察小值
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        return history

