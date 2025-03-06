from audioop import error
import casadi as ca
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Callable


class OptimizationProblemConfiguration:
    """优化问题配置类，用于存储优化问题的各种配置参数"""

    def __init__(self,
                 variables: List[ca.SX],
                 objective_function: ca.SX,
                 equality_constraints: Optional[Dict[str, Any]] = None,
                 inequality_constraints: Optional[List] = None,
                 initial_guess: Optional[Union[List, np.ndarray, ca.DM]] = None):
        """
        初始化优化问题配置

        Args:
            variables: 优化变量列表
            objective_function: 目标函数表达式
            equality_constraints: 等式约束，格式为{"A": A矩阵, "B": B矩阵}，满足 Ax = B
                A和B可以是ca.DM对象、numpy数组或列表，也可以是这些类型的列表
            inequality_constraints: 不等式约束列表，格式为 g(x) <= 0
            initial_guess: 初始猜测值，可以是列表、numpy数组或CasADi DM对象
        """
        # 检查变量和目标函数
        if not variables:
            raise ValueError("变量列表不能为空")
        if not isinstance(objective_function, ca.SX):
            raise TypeError("目标函数必须是CasADi符号表达式")

        self.variables = variables
        self.objective_function = objective_function
        self.num_variables = len(variables)

        # 处理等式约束
        if equality_constraints is not None:
            # 验证等式约束格式
            if not isinstance(equality_constraints, dict):
                raise TypeError("等式约束必须是字典类型")

            if "A" not in equality_constraints or "B" not in equality_constraints:
                raise ValueError("等式约束必须包含'A'和'B'键")

            # 获取A和B数据
            A_data = equality_constraints["A"]
            B_data = equality_constraints["B"]

            # 处理A：识别是单个多行矩阵还是多个矩阵的列表
            if isinstance(A_data, list):
                # A已经是列表，检查是否为空
                if not A_data:
                    A_list = []
                # 检查是否为多行矩阵格式
                elif all(isinstance(row, list) for row in A_data) and all(len(row) == len(variables) for row in A_data):
                    # 这是一个多行矩阵 [[1, 1], [1, -1]] - 应作为单个矩阵处理
                    A_list = [A_data]
                elif isinstance(A_data[0], (list, np.ndarray)) or (
                        hasattr(A_data[0], 'size1') and hasattr(A_data[0], 'size2')):
                    # A是多个矩阵的列表
                    A_list = A_data
                else:
                    # A是单个向量 [1, 1]
                    A_list = [A_data]
            else:
                # A不是列表，包装为列表
                A_list = [A_data]

            # 处理B：确保B与A结构匹配
            if isinstance(B_data, list):
                # B已经是列表
                if not B_data:
                    B_list = []
                elif len(A_list) == 1 and len(B_data) > 1 and not isinstance(B_data[0], (list, np.ndarray)) and not (
                        hasattr(B_data[0], 'size1') and hasattr(B_data[0], 'size2')):
                    # 单矩阵多行约束情况: A是一个多行矩阵，B是对应的向量
                    B_list = [B_data]
                elif isinstance(B_data[0], (list, np.ndarray)) or (
                        hasattr(B_data[0], 'size1') and hasattr(B_data[0], 'size2')):
                    # B是矩阵列表
                    B_list = B_data
                else:
                    # 其他情况 - 根据A的结构决定
                    if len(A_list) == 1:
                        B_list = [B_data]
                    else:
                        B_list = [[b] for b in B_data]  # 每个元素包装为列表
            else:
                # B不是列表，包装为列表
                B_list = [B_data]

            # 检查A和B列表长度是否匹配
            if len(A_list) != len(B_list):
                raise ValueError(f"等式约束A列表长度({len(A_list)})必须等于B列表长度({len(B_list)})")

            # 转换和验证每个约束
            processed_A_list = []
            processed_B_list = []

            for i, (A_item, B_item) in enumerate(zip(A_list, B_list)):
                # 转换A为ca.DM
                if isinstance(A_item, ca.DM):
                    A_dm = A_item
                elif isinstance(A_item, np.ndarray):
                    A_dm = ca.DM(A_item)
                elif isinstance(A_item, list):
                    if not A_item:  # 空列表
                        raise ValueError(f"等式约束A[{i}]不能为空")
                    # 检查是否为嵌套列表
                    if isinstance(A_item[0], list):
                        # 嵌套列表表示矩阵
                        A_dm = ca.DM(A_item)
                    else:
                        # 单层列表表示向量
                        if len(A_item) != self.num_variables:
                            raise ValueError(
                                f"等式约束A[{i}]向量长度({len(A_item)})必须等于变量数量({self.num_variables})")
                        A_dm = ca.DM([A_item])  # 创建1xN矩阵
                else:
                    raise TypeError(f"等式约束A[{i}]类型必须是ca.DM、numpy数组或列表")

                # 转换B为ca.DM
                if isinstance(B_item, ca.DM):
                    B_dm = B_item
                elif isinstance(B_item, np.ndarray):
                    B_dm = ca.DM(B_item)
                elif isinstance(B_item, (list, float, int)):
                    if isinstance(B_item, (float, int)):
                        B_dm = ca.DM([B_item])
                    else:
                        B_dm = ca.DM(B_item)
                else:
                    raise TypeError(f"等式约束B[{i}]类型必须是ca.DM、numpy数组、列表或标量")

                # 确保B是列向量
                if B_dm.is_vector() and B_dm.size2() > 1:
                    B_dm = B_dm.T  # 转置为列向量

                # 处理A的维度
                if A_dm.size2() == 1 and A_dm.size1() == self.num_variables:
                    # A是列向量，但应该是行向量或矩阵
                    A_dm = A_dm.T

                # 确保A的列数等于变量数量
                if A_dm.size2() != self.num_variables:
                    raise ValueError(f"等式约束A[{i}]的列数({A_dm.size2()})必须等于变量数量({self.num_variables})")

                # 确保A的行数等于B的行数
                if A_dm.size1() != B_dm.size1():
                    raise ValueError(f"等式约束A[{i}]的行数({A_dm.size1()})必须等于B[{i}]的行数({B_dm.size1()})")

                processed_A_list.append(A_dm)
                processed_B_list.append(B_dm)

            # 更新等式约束
            self.equality_constraints = {"A": processed_A_list, "B": processed_B_list}
        else:
            # 默认为空等式约束
            self.equality_constraints = {"A": [], "B": []}

        # 处理不等式约束
        if inequality_constraints is not None:
            if not isinstance(inequality_constraints, list):
                inequality_constraints = [inequality_constraints]
            self.inequality_constraints = inequality_constraints
        else:
            self.inequality_constraints = []

        # 处理初始猜测值
        if initial_guess is not None:
            # 转换各种类型为DM对象
            if isinstance(initial_guess, ca.DM):
                if initial_guess.size1() != self.num_variables:
                    raise ValueError(f"初始猜测值维度({initial_guess.size1()})与变量维度({self.num_variables})不匹配")
                self.initial_guess = initial_guess
            elif isinstance(initial_guess, (list, np.ndarray)):
                if len(initial_guess) != self.num_variables:
                    raise ValueError(f"初始猜测值维度({len(initial_guess)})与变量维度({self.num_variables})不匹配")
                self.initial_guess = ca.DM(initial_guess)
            else:
                raise TypeError("初始猜测值必须是list、numpy数组或ca.DM类型")
        else:
            # 如果没有提供初始值，使用零向量
            self.initial_guess = ca.DM.zeros(self.num_variables)

    @classmethod
    def from_dict(cls, configuration: Dict[str, Any]):
        """
        从字典创建配置对象（向后兼容）

        Args:
            configuration: 配置字典

        Returns:
            配置对象
        """
        return cls(
            variables=configuration.get('variables', []),
            objective_function=configuration.get('objective_function', ca.SX.zeros(1)),
            equality_constraints=configuration.get('equality_constraints'),
            inequality_constraints=configuration.get('inequality_constraints', []),
            initial_guess=configuration.get('initial_guess')
        )


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
        self._alpha = 0.01  # 步长
        self._augmented_equality_penalty = 0.2  # 增广拉格朗日惩罚因子
        self._augmented_inequality_penalty = 50
        self.A = None
        self.B = None
        self._max_iterations = 1000  # 最大迭代次数
        self._convergence_tolerance = 1e-6  # 收敛容差

        # 变量相关
        self._initial_guess = None
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

        # 优化历史记录
        self._iteration_history = {
            'x': [],
            'mu': [],
            'objective': [],
            'constraint_violation': []
        }

        # 如果提供了配置，则使用配置初始化
        if configuration is not None:
            self.set_objective_function(configuration.objective_function)
            self.set_variables(configuration.variables)
            for inequality_constraint in configuration.inequality_constraints:
                self.add_inequality_constraint(inequality_constraint)
            if configuration.equality_constraints is not None and configuration.equality_constraints["A"]:
                A_list = configuration.equality_constraints["A"]
                B_list = configuration.equality_constraints["B"]

                # 组合多个约束时的矩阵构建
                if len(A_list) > 1:
                    # 垂直堆叠所有A矩阵和B向量
                    self.A = ca.vertcat(*A_list)
                    self.B = ca.vertcat(*B_list)
                else:
                    self.A = A_list[0]
                    self.B = B_list[0]

                # 对每个A矩阵和B向量
                for A_matrix, B_vector in zip(A_list, B_list):
                    # 检查维度
                    num_constraints = A_matrix.size1()

                    # 对矩阵的每一行添加一个约束
                    for i in range(num_constraints):
                        # 提取第i行
                        A_row = A_matrix[i, :]  # 获取第i行
                        B_value = B_vector[i]  # 获取第i个B值

                        # 添加约束: A_row * x - B_value = 0 (标量表达式)
                        self.add_equality_constraint(ca.mtimes(A_row, self._xs) - B_value)

            self.set_initial_guess(configuration.initial_guess)
            self._use_configuration = True
            self._objective_expression_set = True
            self._variable_defined = True

    def set_solver_parameters(self, alpha: float = None, equality_penalty: float = None,
                              inequality_penalty: float = None, max_iterations: int = None,
                              convergence_tolerance: float = None):
        """
        设置求解器参数

        Args:
            alpha: 步长
            equality_penalty: 等式约束增广拉格朗日惩罚因子
            inequality_penalty: 不等式约束惩罚因子
            max_iterations: 最大迭代次数
            convergence_tolerance: 收敛容差
        """
        if alpha is not None:
            if alpha <= 0:
                raise ValueError("步长必须为正数")
            self._alpha = alpha
        if equality_penalty is not None:
            if equality_penalty <= 0:
                raise ValueError("等式约束惩罚因子必须为正数")
            self._augmented_equality_penalty = equality_penalty
        if inequality_penalty is not None:
            if inequality_penalty <= 0:
                raise ValueError("不等式约束惩罚因子必须为正数")
            self._augmented_inequality_penalty = inequality_penalty
        if max_iterations is not None:
            if max_iterations <= 0:
                raise ValueError("最大迭代次数必须为正整数")
            self._max_iterations = max_iterations
        if convergence_tolerance is not None:
            if convergence_tolerance <= 0:
                raise ValueError("收敛容差必须为正数")
            self._convergence_tolerance = convergence_tolerance

        # 如果已经生成了对偶上升函数，需要重新生成
        if self._dual_ascent_function_generated:
            self._dual_ascent_function_generated = False
            self.generate_dual_ascent_function()

    def dual_ascent(self, step_num: int = None, use_augmented_lagrange_function: bool = False,
                    callback: Callable = None, verbose: bool = False) -> Dict[str, ca.DM]:
        """
        对偶上升法求解优化问题

        Args:
            step_num: 迭代步数，如果为None则使用最大迭代次数
            use_augmented_lagrange_function: 是否使用增广拉格朗日函数
            callback: 每次迭代后调用的回调函数，签名为callback(iteration, x, mu, objective, constraint_violation)
            verbose: 是否打印迭代过程信息

        Returns:
            优化结果字典，包含优化变量、拉格朗日乘子、目标函数值、约束违反度和收敛状态
        """
        if step_num is None:
            step_num = self._max_iterations

        self._use_augmented_lagrange_function = use_augmented_lagrange_function
        self.generate_dual_ascent_function()
        self.set_initial_guess()

        # 清空历史记录
        self._iteration_history = {
            'x': [],
            'mu': [],
            'objective': [],
            'constraint_violation': []
        }

        has_equality = self.has_equality_constraints()

        x_old = None
        mu_old = None
        converged = False
        final_iteration = 0

        for i in range(step_num):
            # 保存旧值用于检查收敛性
            x_old = self._x_k if i > 0 else None
            if has_equality:
                mu_old = self._mu_k if i > 0 else None

            # 执行迭代步骤
            if has_equality:
                self._x_k = self._next_x_function(self._x_k, self._mu_k)
                self._mu_k = self._next_multiplier_function(self._x_k, self._mu_k)
            else:
                self._x_k = self._next_x_function(self._x_k)

            # 计算目标函数值和约束违反度
            obj_value = float(self._objective_function(self._x_k))
            constraint_violation = self._calculate_constraint_violation(self._x_k)

            # 记录历史
            self._iteration_history['x'].append(self._x_k)
            if has_equality:
                self._iteration_history['mu'].append(self._mu_k)
            self._iteration_history['objective'].append(obj_value)
            self._iteration_history['constraint_violation'].append(constraint_violation)

            # 调用回调函数
            if callback is not None:
                callback(i, self._x_k, self._mu_k if has_equality else None, obj_value, constraint_violation)

            # 打印进度
            if verbose and (i % 10 == 0 or i == step_num - 1):
                print(f"迭代 {i}: 目标函数值 = {obj_value:.6f}, 约束违反度 = {constraint_violation:.6f}")

            # 检查收敛性
            if x_old is not None:
                x_diff = float(ca.norm_2(self._x_k - x_old))
                mu_diff = float(ca.norm_2(self._mu_k - mu_old)) if has_equality else 0

                if x_diff < self._convergence_tolerance and mu_diff < self._convergence_tolerance:
                    converged = True
                    final_iteration = i
                    if verbose:
                        print(f"迭代 {i}: 收敛，x差值 = {x_diff:.6e}, mu差值 = {mu_diff:.6e}")
                    break

        # 返回结果
        result = {
            'x': self._x_k,
            'objective': obj_value,
            'constraint_violation': constraint_violation,
            'converged': converged,
            'iterations': final_iteration + 1 if converged else step_num
        }
        if has_equality:
            result['mu'] = self._mu_k

        return result

    def set_variables(self, variables: List[ca.SX]):
        """
        设置优化变量

        Args:
            variables: 优化变量列表
        """
        if not variables:
            raise ValueError("变量列表不能为空")

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

        if not self._variable_defined:
            raise ValueError("未设置变量，无法生成对偶上升函数")

        if not self._objective_expression_set:
            raise ValueError("未设置目标函数，无法生成对偶上升函数")

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
        has_equality = self.has_equality_constraints()

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
        if not isinstance(equality_constraint, ca.SX):
            raise TypeError("等式约束必须是CasADi符号表达式")

        self._equality_constraints.append(equality_constraint)
        # 添加约束后需要重新生成对偶上升函数
        self._dual_ascent_function_generated = False
        self._multiplier_defined = False  # 需要重新定义乘子

    def get_xs(self):
        """
        获取优化变量

        Returns:
            优化变量符号表达式
        """
        if not self._variable_defined:
            raise ValueError("未设置变量")
        return self._xs

    def add_inequality_constraint(self, inequality_constraint: ca.SX):
        """
        添加不等式约束 g(x) <= 0， 作用到拉格朗日函数上，用惩罚代替不等式约束，提高求解的效率

        Args:
            inequality_constraint: 不等式约束表达式
        """
        if not isinstance(inequality_constraint, ca.SX):
            raise TypeError("不等式约束必须是CasADi符号表达式")

        self._objective_expression += (self._augmented_inequality_penalty / 2) * ca.fmax(0, inequality_constraint) ** 2
        self._inequality_constraints.append(inequality_constraint)
        # 添加约束后需要更新目标函数
        if self._variable_defined:
            self._objective_function = ca.Function('objective_function',
                                                   [self._xs],
                                                   [self._objective_expression])
        # 需要重新生成对偶上升函数
        self._dual_ascent_function_generated = False

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

                # 初始化数值乘子
                if self._mu_k is None:
                    self._mu_k = ca.DM.zeros(len(self._equality_constraints))
            else:
                self._mus = None  # 如果没有等式约束，设置为None
                self._mu_k = None
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
        if not isinstance(objective_expression, ca.SX):
            raise TypeError("目标函数必须是CasADi符号表达式")

        self._objective_expression = objective_expression
        self._objective_expression_set = True

        # 如果已经定义了变量，更新目标函数
        if self._variable_defined:
            self._objective_function = ca.Function('objective_function',
                                                   [self._xs],
                                                   [self._objective_expression])
        # 需要重新生成对偶上升函数
        self._dual_ascent_function_generated = False

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
        return {'equality': self._mus}

    def get_initial_guess(self):
        """
        获取初始猜测值

        Returns:
            优化变量的初始值
        """
        if not self._initial_guess_set:
            raise ValueError("未设置初始猜测值")
        return self._initial_guess

    def set_initial_guess(self, initial_guess: Optional[ca.DM] = None):
        """
        设置优化变量和乘子的初始猜测值

        Args:
            initial_guess: 优化变量的初始值，如果为None则使用零向量

        Raises:
            ValueError: 如果未设置变量或未生成拉格朗日函数
        """
        if initial_guess is not None:
            if not self._variable_defined:
                raise ValueError("未设置变量，无法设置初始猜测值")

            if initial_guess.size1() != self._xs.size1():
                raise ValueError(f"初始猜测值维度 ({len(initial_guess)}) 与变量维度 ({self._xs.size1()}) 不匹配")

            self._initial_guess = ca.DM(initial_guess)
            self._x_k = initial_guess
        else:
            if self._variable_defined:
                self._x_k = ca.DM.zeros(self._xs.size1())
                self._initial_guess = self._x_k
            else:
                raise ValueError('未设置变量，无法设置初始猜测值')

            # 初始化乘子
            self.initialize_multipliers()

        self._initial_guess_set = True

    def set_mu_and_lambda(self, mu_: ca.DM, lambda_: ca.DM = None):
        """
        设置拉格朗日乘子的值

        Args:
            mu_: 等式约束乘子的值
            lambda_: 不等式约束乘子的值（当前未使用）
        """
        if not self._multiplier_defined:
            self.initialize_multipliers()

        if self._mus is not None:
            if mu_.size1() != self._mus.size1():
                raise ValueError(f"等式约束乘子维度 ({mu_.size1()}) 与约束数量 ({self._mus.size1()}) 不匹配")
            self._mu_k = mu_
        else:
            if mu_ is not None:
                import warnings
                warnings.warn('没有等式约束，设置等式乘子无效')

    def has_inequality_constraints(self) -> bool:
        """
        检查是否有不等式约束

        Returns:
            是否有不等式约束
        """
        return len(self._inequality_constraints) > 0

    def has_equality_constraints(self) -> bool:
        """
        检查是否有等式约束

        Returns:
            是否有等式约束
        """
        return len(self._equality_constraints) > 0

    def compute_next_x(self, x: ca.DM, mu_: ca.DM) -> Dict[str, ca.DM]:
        """
        计算下一步的优化变量和乘子值

        Args:
            x: 当前优化变量值
            mu_: 当前等式约束乘子值

        Returns:
            包含更新后的优化变量和乘子的字典
        """
        # 更新当前值
        self._x_k = x
        self._mu_k = mu_

        # 确保已生成对偶上升函数
        self.generate_dual_ascent_function()

        has_equality = self.has_equality_constraints()

        # 执行迭代步骤
        if has_equality:
            next_x = self._next_x_function(self._x_k, self._mu_k)
            next_mu = self._next_multiplier_function(self._x_k, self._mu_k)
        else:
            next_x = self._next_x_function(self._x_k)
            next_mu = None

        # 返回结果
        result = {'x': next_x}
        if has_equality:
            result['mu'] = next_mu
        return result

    def _calculate_constraint_violation(self, x: ca.DM) -> float:
        """
        计算约束违反度

        Args:
            x: 当前优化变量值

        Returns:
            约束违反度（等式约束和不等式约束违反度的最大值）
        """
        eq_violation = 0
        ineq_violation = 0

        # 计算等式约束违反度
        if self.has_equality_constraints():
            for h in self._equality_constraints:
                h_func = ca.Function('h', [self._xs], [h])
                eq_violation = max(eq_violation, abs(float(h_func(x))))

        # 计算不等式约束违反度
        if self.has_inequality_constraints():
            for g in self._inequality_constraints:
                g_func = ca.Function('g', [self._xs], [g])
                g_val = float(g_func(x))
                if g_val > 0:
                    ineq_violation = max(ineq_violation, g_val)

        return max(eq_violation, ineq_violation)

    def get_iteration_history(self) -> Dict[str, List]:
        """
        获取迭代历史记录

        Returns:
            包含优化变量、乘子、目标函数值和约束违反度的历史记录
        """
        return self._iteration_history

    @property
    def xs(self):
        """
        获取优化变量（符号表达式）

        Returns:
            优化变量符号表达式
        """
        if not self._variable_defined:
            raise ValueError("未设置变量")
        return self._xs
